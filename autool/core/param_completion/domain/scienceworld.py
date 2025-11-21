from autool.core.tool_predict.datastruct import ToolGraph
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict
import re
import os
import random # 用于随机选择
from ..param_dependency import ExecutionHistory
from collections import defaultdict
import json
from ..param_completion import EnvironmentAdapter, ParameterFillingFramework
from autool.utils.parser.scienceworld import parse_scienceworld_action, infer_output_from_observation


class ScienceWorldAdapter(EnvironmentAdapter):
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.goal: Optional[str] = None
        self.state = self._get_initial_state()
        self.action_parser_fn = parse_scienceworld_action
        self.output_infer_fn = infer_output_from_observation

    def _get_initial_state(self) -> Dict:
        return {
            "current_location": None,
            "inventory": set(),
            "object_properties": defaultdict(dict),  # {"obj_id": {"state": "open", "temp": 10}}
            "visible_objects": set(), # Objects explicitly listed by look_around in current_location
            "known_objects_globally": set(), # All objects ever mentioned/seen
            "container_contents": defaultdict(set), # {"container_id": {"obj1", "obj2"}}
            "last_observations": defaultdict(lambda: None),
            "known_locations": set(),
        }

    def reset(self, init_observation: str = None, goal: Optional[str] = None) -> None:
        self.goal = goal
        self.state = self._get_initial_state()
        if init_observation:
            # Attempt to parse initial observation to set current location and visible objects
            action_parsed = {"tool_name": "look_around", "inputs": {}} # Simulate a look_around
            structured_output = self.infer_output(action_parsed, init_observation)
            self._update_state_from_parsed_observation(action_parsed, structured_output)


    def parse_action(self, action_text: str, tool_descriptions: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        return self.action_parser_fn(action_text, tool_descriptions)

    def infer_output(self, tool_name: str, inputs: Dict[str, Any], result: Any) -> Dict[str, Any]:
        parsed = infer_output_from_observation({"tool_name": tool_name, "inputs": inputs}, result)
        return parsed

    def _update_state_from_parsed_observation(self, action_parsed: Dict[str, Any], structured_output: Dict[str, Any]):
        """Helper to update state based on structured_output, separated for reuse (e.g. in reset)"""
        tool_name = action_parsed.get("tool_name")
        inputs = action_parsed.get("inputs", {})

        if structured_output.get("status") != "success":
            return # Don't update state on failure

        # Update current location from go_to or look_around
        if tool_name == "go_to" and "new_location" in structured_output:
            self.state["current_location"] = structured_output["new_location"]
            self.state["known_locations"].add(structured_output["new_location"])
            self.state["visible_objects"].clear() # Objects in old location no longer visible
        elif tool_name == "look_around" and "current_room" in structured_output:
            if self.state["current_location"] != structured_output["current_room"]:
                 self.state["visible_objects"].clear() # Moved implicitly or first look
            self.state["current_location"] = structured_output["current_room"]
            self.state["known_locations"].add(structured_output["current_room"])
        
        # Update inventory
        if tool_name == "pick_up" and "picked_object" in structured_output:
            obj = structured_output["picked_object"]
            self.state["inventory"].add(obj)
            self.state["visible_objects"].discard(obj) # No longer in location if picked up
            self.state["known_objects_globally"].add(obj)
            # Remove from any container it might have been in at current location
            if self.state["current_location"]:
                for container, items in self.state["container_contents"].items():
                    if container in self.state["visible_objects"] or self.state["object_properties"].get(container, {}).get("location") == self.state["current_location"]:
                        items.discard(obj)

        if tool_name == "put_down" and "put_object" in structured_output:
            obj = structured_output["put_object"]
            self.state["inventory"].discard(obj)
            self.state["visible_objects"].add(obj) # Now visible in location
            self.state["known_objects_globally"].add(obj)
            if self.state["current_location"]:
                 self.state["object_properties"][obj]["location"] = self.state["current_location"]


        # Update from look_around
        if tool_name == "look_around":
            if "objects" in structured_output:
                current_loc_objs = set(structured_output["objects"])
                self.state["visible_objects"] = current_loc_objs
                self.state["known_objects_globally"].update(current_loc_objs)
                for obj_name in current_loc_objs:
                    self.state["object_properties"][obj_name]["location"] = self.state["current_location"]

            if "doors" in structured_output:
                for door_info in structured_output["doors"]:
                    door_name = door_info["name"]
                    self.state["object_properties"][door_name]["state"] = door_info["state"]
                    self.state["object_properties"][door_name]["is_door"] = True
                    self.state["known_objects_globally"].add(door_name)
                    # Infer connected location from door name "door to the X"
                    match_loc = re.search(r"door to the (.*)", door_name, re.IGNORECASE)
                    if match_loc:
                        self.state["known_locations"].add(match_loc.group(1))


        # Update from look_at / look_in
        if tool_name == "look_at" and "object_description" in structured_output:
            obj_looked_at = inputs.get("OBJ")
            if obj_looked_at:
                self.state["known_objects_globally"].add(obj_looked_at)
                if "state" in structured_output: # e.g. "closed", "open"
                    self.state["object_properties"][obj_looked_at]["state"] = structured_output["state"]
                # Could parse more properties from description if needed

        if tool_name == "look_in" and "contains" in structured_output:
            container_looked_in = inputs.get("OBJ_container")
            if container_looked_in:
                self.state["known_objects_globally"].add(container_looked_in)
                self.state["container_contents"][container_looked_in] = set(structured_output["contains"])
                self.state["known_objects_globally"].update(structured_output["contains"])
                for item_in_container in structured_output["contains"]:
                     self.state["object_properties"][item_in_container]["location"] = container_looked_in


        # Update from open/close/activate/deactivate
        if tool_name in ["open", "close", "activate", "deactivate"] and "new_state" in structured_output:
            obj_changed = structured_output.get("object_changed", inputs.get("OBJ"))
            if obj_changed:
                self.state["known_objects_globally"].add(obj_changed)
                self.state["object_properties"][obj_changed]["state"] = structured_output["new_state"]
        
        # Update from 'use thermometer'
        if tool_name == "use" and "temperature_reading" in structured_output:
            measured_obj_name = structured_output.get("measured_object", inputs.get("OBJ_target"))
            if measured_obj_name: # OBJ_target might be "substance in X"
                self.state["known_objects_globally"].add(measured_obj_name)
                self.state["object_properties"][measured_obj_name]["temperature"] = structured_output["temperature_reading"]
        
        # Update from inventory action
        if tool_name == "inventory" and "items" in structured_output:
            self.state["inventory"] = set(structured_output["items"])
            self.state["known_objects_globally"].update(self.state["inventory"])

    def update_state(self, action_parsed: Dict[str, Any], structured_output: Dict[str, Any]) -> None:
        try:
            tool_name = action_parsed.get("tool_name")
            if not tool_name:
                return
            
            # Store last observation for this tool type
            self.state["last_observations"][tool_name] = structured_output 
            
            self._update_state_from_parsed_observation(action_parsed, structured_output)

            if self.debug:
                print(f"  State after '{action_parsed.get('raw_content', tool_name)}':")
                print(f"    Goal: {self.goal}")
                print(f"    Current Loc: {self.state['current_location']}")
                print(f"    Inventory: {self.state['inventory']}")
                print(f"    Visible Objs: {self.state['visible_objects']}")
                # print(f"    Obj Props: {dict(self.state['object_properties'])}") # Can be verbose
                # print(f"    Container Contents: {dict(self.state['container_contents'])}") # Can be verbose
                print(f"    Known Locations: {self.state['known_locations']}")


        except Exception as e:
            if self.debug:
                print(f"Error updating ScienceWorld state: {e}")

    def get_contextual_params(self, action_type: str, missing_params: Set[str], required_params_info: Dict) -> Dict[str, Any]:
        filled = {}
        if self.debug: print(f"  Phase 3 (ScienceWorld): Checking Env State for {missing_params} for action '{action_type}'...")
        
        current_loc = self.state["current_location"]
        inventory = self.state["inventory"]
        visible_objs = self.state["visible_objects"] # Objects directly in current room after look_around
        obj_props = self.state["object_properties"]
        container_contents = self.state["container_contents"]
        known_globally = self.state["known_objects_globally"]
        
        # Goal related objects/locations (simple extraction)
        goal_objects_explicit = set()
        goal_locations_explicit = set()
        if self.goal:
            # Try to find known objects/locations mentioned in goal
            for item in known_globally.union(self.state["known_locations"]):
                if item and item.lower() in self.goal.lower():
                    if item in self.state["known_locations"] or "door to" in item or any(k in item.lower() for k in ["kitchen", "hallway", "outside", "greenhouse"]): # Heuristic for location
                        goal_locations_explicit.add(item)
                    else:
                        goal_objects_explicit.add(item)
        
        if self.debug:
            print(f"    Context: GoalObjs={goal_objects_explicit}, GoalLocs={goal_locations_explicit}")
            print(f"             CurLoc={current_loc}, Inv={inventory}, Visible={visible_objs}")

        for param_name in list(missing_params): # Iterate over a copy
            param_info = required_params_info.get(param_name, {})
            # param_type = param_info.get("type", "str") # Not strictly used for selection yet
            value_found = None
            
            # Simplified logic:
            # OBJ: For actions like pick_up, open, close, look_at, use OBJ_tool, eat, focus_on
            if param_name.upper() == "OBJ" or param_name.upper() == "OBJ_TOOL" or param_name.upper() == "OBJ_TARGET":
                candidates = set()
                # Priority 1: Goal-mentioned objects visible or in inventory
                candidates.update(goal_objects_explicit.intersection(visible_objs.union(inventory)))
                
                # Priority 2: Other visible objects not yet "stable" (e.g. not doors if not opening/closing)
                if not candidates:
                    candidates.update(o for o in visible_objs if not obj_props.get(o, {}).get("is_door", False) or action_type in ["open", "close"])
                
                # Priority 3: Other inventory items
                if not candidates and action_type not in ["pick_up"]: # don't pick up from inventory
                    candidates.update(inventory - goal_objects_explicit)

                # Priority 4 (for 'use OBJ_target'): if OBJ_tool is known, what can it be used on? (complex, skip for now)
                
                # Filter by action needs
                if action_type == "pick_up":
                    candidates = candidates - inventory # Can't pick up if already in inventory
                elif action_type == "open":
                    candidates = {o for o in candidates if obj_props.get(o, {}).get("state") != "open"}
                elif action_type == "close":
                    candidates = {o for o in candidates if obj_props.get(o, {}).get("state") == "open"}
                
                if candidates:
                    value_found = random.choice(list(candidates))

            # LOC: For go_to
            elif param_name.upper() == "LOC":
                candidates = set()
                 # Priority 1: Goal-mentioned locations
                candidates.update(goal_locations_explicit)
                # Priority 2: Other known locations not current
                if not candidates:
                    candidates.update(self.state["known_locations"] - {current_loc})
                if candidates:
                    value_found = random.choice(list(candidates))
            
            # OBJ_container: for look_in, pour, dunk
            elif param_name.upper() == "OBJ_CONTAINER" or param_name.upper() == "OBJ_SOURCE_CONTAINER" or param_name.upper() == "OBJ_TARGET_CONTAINER" or param_name.upper() == "OBJ_DESTINATION" or param_name.upper() == "OBJ_LIQUID_CONTAINER":
                # General containers: visible objects that are known containers or goal-mentioned containers
                # This needs better type information for objects (is_container?)
                # For now, just use visible objects as candidates
                candidates = set()
                candidates.update(goal_objects_explicit.intersection(visible_objs.union(inventory))) # goal items that might be containers
                if not candidates:
                    candidates.update(o for o in visible_objs if "cup" in o or "pot" in o or "jar" in o or "bowl" in o or "sink" in o or "freezer" in o or "fridge" in o or "cupboard" in o or "drawer" in o) # Heuristic
                if not candidates:
                    candidates.update(inventory) # if pouring from inventory

                if candidates:
                    value_found = random.choice(list(candidates))

            if value_found:
                filled[param_name] = value_found
                missing_params.remove(param_name)
                if self.debug: print(f"      Inferred '{param_name}' from ScienceWorld env state = {value_found}")
        
        return filled

    def generate_action_from_params(self, action_type: str, params: Dict[str, Any]) -> str:
        """Generates a ScienceWorld action string from tool name and parameters."""
        # Based on scienceworld_base.json format
        # Order of params might matter for some tools, this is a simple joiner
        
        # Handle specific tool structures first
        if action_type == "task" or action_type == "inventory":
            return action_type
        if action_type == "check":
            return "check valid actions"
        
        if action_type == "look_around":
            return "look around"
        
        if action_type == "wait":
            return f"wait"

        parts = [action_type.replace("_", " ")] # e.g. pick_up -> pick up

        # Simplified logic: append params. A more robust version would use tool_descriptions for order and keywords.
        obj = params.get("OBJ") or params.get("OBJ_tool") # Common primary object
        loc = params.get("LOC")
        
        # --- 去除数量词 a ---
        if obj and obj.startswith("a "):
            obj = obj[2:]

        obj_to_move = params.get("OBJ_to_move")
        obj_destination = params.get("OBJ_destination")
        
        obj_source_container = params.get("OBJ_source_container")
        obj_target_container = params.get("OBJ_target_container")

        obj_to_dunk = params.get("OBJ_to_dunk")
        obj_liquid_container = params.get("OBJ_liquid_container")

        obj_target_for_use = params.get("OBJ_target") # For 'use OBJ_tool on OBJ_target'
        
        obj_container_for_look_in = params.get("OBJ_container")


        if action_type == "go_to" and loc:
            parts.append(loc)
        elif action_type in ["open", "close", "pick_up", "put_down", "mix", "read", "activate", "deactivate", "eat", "flush", "focus_on", "look_at"] and obj:
            parts.append(obj)
        elif action_type == "look_in" and obj_container_for_look_in:
            parts.append(obj_container_for_look_in)
        elif action_type == "move" and obj_to_move and obj_destination:
            parts.append(obj_to_move)
            parts.append("to")
            parts.append(obj_destination)
        elif action_type == "pour" and obj_source_container and obj_target_container:
            parts.append(obj_source_container)
            parts.append("into")
            parts.append(obj_target_container)
        elif action_type == "dunk" and obj_to_dunk and obj_liquid_container:
            parts.append(obj_to_dunk)
            parts.append("into")
            parts.append(obj_liquid_container)
        elif action_type == "use" and obj: # obj here is OBJ_tool
            parts.append(obj)
            if obj_target_for_use:
                parts.append("on")
                parts.append(obj_target_for_use)
        elif obj: # Fallback for single OBJ param tools if not caught above
             parts.append(obj)
        # Else, if params is not empty but didn't match specific structures, just join them (less ideal)
        elif params:
            for p_val in params.values():
                if p_val is not None: parts.append(str(p_val))
            
        return " ".join(parts)

    def _check_type(self, value: Any, expected_type: str) -> bool:
        # Simple type check, can be expanded
        if expected_type == "int":
            return isinstance(value, int)
        elif expected_type == "str":
            return isinstance(value, str)
        elif expected_type == "float":
            return isinstance(value, float)
        # Add more types if needed
        return True # Default to true if type is not strictly checked or unknown


class ScienceWorldParamCompletion:
    def __init__(self,
                 tool_graph: ToolGraph,
                #  tool_descriptions: Dict[str, Dict],
                 dependency_graph_path: Optional[str] = None,
                 history_max_len: int = 20, # ScienceWorld might need longer history
                 goal: Optional[str] = None,
                 debug: bool = False):
        self.debug = debug
        self.adapter = ScienceWorldAdapter(debug=debug)
        if goal:
            self.adapter.goal = goal

        self.framework = ParameterFillingFramework(
            tool_graph=tool_graph,
            # tool_descriptions=tool_descriptions,
            adapter=self.adapter,
            param_dependency_path=dependency_graph_path,
            history_max_len=history_max_len,
            debug=debug
        )

    def record_execution(self, action_text: str, result: Any, goal: Optional[str] = None):
        if goal and self.adapter.goal != goal: # Update goal if changed
            self.adapter.goal = goal
            if self.debug: print(f"Goal updated in ScienceWorldAdapter: {goal}")

        action_parsed = self.adapter.parse_action(action_text)
        if not action_parsed:
            if self.debug: print(f"Could not parse action: {action_text}")
            # Record a dummy/error entry if parsing fails? Or skip? For now, skip.
            return
        
        # Store raw action text in parsed dict if not already there (for state update debug)
        if "raw_content" not in action_parsed:
            action_parsed["raw_content"] = action_text

        self.framework.record_execution(
            action_text=action_text, # Pass original text for history
            action_parsed=action_parsed, # Pass structured action
            result=result
        )
    
    def fill_parameters(self, target_tool: str, existing_params: Optional[Dict[str, Any]] = None, lookback_k: int = 5, goal: Optional[str] = None) -> Dict[str, Any]:
        if goal and self.adapter.goal != goal: # Update goal if changed
            self.adapter.goal = goal
            if self.debug: print(f"Goal updated in ScienceWorldAdapter for fill_parameters: {goal}")
            
        return self.framework.fill_parameters(
            target_tool=target_tool,
            existing_params=existing_params,
            lookback_k=lookback_k
        )

    def generate_action_from_params(self, action_type: str, params: Dict[str, Any]) -> str:
        return self.adapter.generate_action_from_params(action_type, params)

    def set_debug(self, debug: bool):
        self.debug = debug
        self.adapter.debug = debug
        self.framework.set_debug(debug)
        
    def reset_history_and_state(self, init_observation: str = None, goal: Optional[str] = None):
        self.framework.reset_history()
        self.adapter.reset(init_observation, goal)
        if self.debug: print("ScienceWorld history and state reset.")


if __name__ == '__main__':
    # Example Usage (Illustrative)
    print("ScienceWorld Adapter and ParamCompletion Framework")
    # Mock data for demonstration
    mock_tool_graph = ToolGraph() # Assume it's populated
    mock_tool_descriptions = {
        "go_to": {"tool": "go_to", "args": [{"arg_name": "LOC", "arg_type": "str"}]},
        "pick_up": {"tool": "pick_up", "args": [{"arg_name": "OBJ", "arg_type": "str"}]},
        "look_around": {"tool": "look_around", "args": []},
        # ... other tools
    }
    
    # Test action parsing
    print("\n--- Action Parsing Test ---")
    actions_to_test = [
        "go to kitchen",
        "pick up metal pot",
        "use thermometer on water",
        "look around",
        "wait",
        "wait 3",
        "focus on hatchling giant tortoise"
    ]
    for act in actions_to_test:
        parsed = parse_scienceworld_action(act)
        print(f"'{act}' -> {parsed}")

    # Test observation inference
    print("\n--- Observation Inference Test ---")
    adapter_test = ScienceWorldAdapter(debug=True)
    
    obs_look_around_kitchen = "This room is called the kitchen. In it, you see: \\n\\ta substance called soap\\n\\ta painting\\n\\ta thermometer, currently reading a temperature of 10 degrees celsius\\n\\ta counter. On the counter is: a bowl (containing a banana, a potato, a red apple, an orange), a drawer.\\n\\ta sink, which is turned off. In the sink is: nothing.\\nYou also see:\\n\\tA door to the outside (that is open)\\n\\tA door to the bathroom (that is open)"
    parsed_act_look = parse_scienceworld_action("look around")
    inferred_look = adapter_test.infer_output(parsed_act_look, obs_look_around_kitchen)
    print(f"Look Around Obs: {inferred_look}")
    adapter_test.update_state(parsed_act_look, inferred_look)

    obs_go_outside = "You move to the outside."
    parsed_act_go = parse_scienceworld_action("go to outside")
    inferred_go = adapter_test.infer_output(parsed_act_go, obs_go_outside)
    print(f"Go To Obs: {inferred_go}")
    adapter_test.update_state(parsed_act_go, inferred_go)
    
    obs_look_around_outside = "This outside location is called the outside. Here you see: \\n\\tthe agent\\n\\ta substance called air\\n\\tan axe\\n\\ta crocodile egg\\nYou also see:\\n\\tA door to the foundry (that is open)"
    inferred_look_outside = adapter_test.infer_output(parsed_act_look, obs_look_around_outside)
    print(f"Look Around Outside Obs: {inferred_look_outside}")
    adapter_test.update_state(parsed_act_look, inferred_look_outside)

    obs_pick_axe = "You move the axe to the inventory."
    parsed_act_pick = parse_scienceworld_action("pick up axe")
    inferred_pick = adapter_test.infer_output(parsed_act_pick, obs_pick_axe)
    print(f"Pick Up Obs: {inferred_pick}")
    adapter_test.update_state(parsed_act_pick, inferred_pick)

    # Setup for param completion
    # sw_param_completer = ScienceWorldParamCompletion(
    #     tool_graph=mock_tool_graph,
    #     tool_descriptions=mock_tool_descriptions,
    #     goal="Find the animal with the longest life span.",
    #     debug=True
    # )
    # Initial observation for reset
    # init_obs_example = "This room is called the workshop. In it, you see: \\n\\tthe agent\\n\\ta substance called air\\n\\ta table. On the table is: a battery, a black wire, a blue wire, a orange wire, a red light bulb, which is off, a switch, which is off, a violet light bulb, which is off, a yellow light bulb, which is off.\\n\\ta ultra low temperature freezer. The ultra low temperature freezer door is closed. \\nYou also see:\\n\\tA door to the hallway (that is open)\\nIn your inventory, you see:\\n\\tan orange"
    # sw_param_completer.reset_history_and_state(init_observation=init_obs_example, goal="Find the animal with the longest life span.")

    # Simulate some history
    # sw_param_completer.record_execution("go to hallway", "You move to the hallway.")
    # sw_param_completer.record_execution("look around", "This room is called the hallway. In it, you see: ... A door to the kitchen (that is open)...")
    # sw_param_completer.record_execution("go to kitchen", "You move to the kitchen.")
    # sw_param_completer.record_execution("look around", "This room is called the kitchen. In it, you see: ... a metal pot ...")

    # Try to fill params
    # filled_params = sw_param_completer.fill_parameters(target_tool="pick_up")
    # print(f"Filled params for 'pick_up': {filled_params}")
    # if filled_params.get("OBJ"):
    #     action_str = sw_param_completer.generate_action_from_params("pick_up", filled_params)
    #     print(f"Generated action: {action_str}")

    # filled_params_goto = sw_param_completer.fill_parameters(target_tool="go_to", goal="Boil water in the kitchen.")
    # print(f"Filled params for 'go_to': {filled_params_goto}")
    # if filled_params_goto.get("LOC"):
    #     action_str_goto = sw_param_completer.generate_action_from_params("go_to", filled_params_goto)
    #     print(f"Generated action: {action_str_goto}")

