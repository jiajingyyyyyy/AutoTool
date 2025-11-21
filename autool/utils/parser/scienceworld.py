import re
from typing import Dict, Any, Optional, List

def _parse_general_one_arg_action(action_text_lower: str, prefix: str, arg_name: str) -> Optional[Dict[str, Any]]:
    """Helper to parse actions with one argument after a prefix."""
    if action_text_lower.startswith(prefix):
        target = action_text_lower[len(prefix):].strip()
        if target:
            # Convert prefix to tool_name (e.g., "pick up " -> "pick_up")
            tool_name = prefix.strip().replace(" ", "_")
            return {"tool_name": tool_name, "inputs": {arg_name: target}}
    return None

def _parse_general_two_arg_action(action_text_lower: str, tool_name: str, pattern: str, arg_names: List[str]) -> Optional[Dict[str, Any]]:
    """Helper to parse actions with two arguments using regex."""
    match = re.match(pattern, action_text_lower)
    if match:
        inputs = {}
        for i, arg_name in enumerate(arg_names):
            inputs[arg_name] = match.group(i+1).strip()
        return {"tool_name": tool_name, "inputs": inputs}
    return None

def parse_scienceworld_action(action_text: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parses a ScienceWorld natural language action into a structured dictionary.
    """
    if not action_text:
        return None

    action_text_lower = action_text.lower().strip()
    # Default structure for unknown actions
    parsed_action = {"tool_name": "unknown", "inputs": {}}

    # Information & Simple Actions
    if action_text_lower == "task":
        return {"tool_name": "task", "inputs": {}}
    if action_text_lower == "inventory":
        return {"tool_name": "inventory", "inputs": {}}
    if action_text_lower == "look around":
        return {"tool_name": "look_around", "inputs": {}}
    if action_text_lower in "check valid actions":
        return {"tool_name": "check", "inputs": {}}
    if action_text_lower == "wait":
        return {"tool_name": "wait", "inputs": {}}
    # Movement
    action = _parse_general_one_arg_action(action_text_lower, "go to ", "LOC")
    if action: return action

    # Inspection
    action = _parse_general_one_arg_action(action_text_lower, "look at ", "OBJ")
    if action: return action
    action = _parse_general_one_arg_action(action_text_lower, "look in ", "OBJ_container")
    if action: return action
    action = _parse_general_one_arg_action(action_text_lower, "read ", "OBJ_readable")
    if action: return action

    # Manipulation
    action = _parse_general_one_arg_action(action_text_lower, "open ", "OBJ")
    if action: return action
    action = _parse_general_one_arg_action(action_text_lower, "close ", "OBJ")
    if action: return action
    action = _parse_general_one_arg_action(action_text_lower, "pick up ", "OBJ")
    if action: return action
    action = _parse_general_one_arg_action(action_text_lower, "put down ", "OBJ")
    if action: return action
    action = _parse_general_one_arg_action(action_text_lower, "mix ", "OBJ_container") # OBJ is the container being mixed
    if action: return action
    
    action = _parse_general_two_arg_action(action_text_lower, "move", r"move (.+?) to (.+)", ["OBJ_source", "OBJ_target"])
    if action: return action
    action = _parse_general_two_arg_action(action_text_lower, "pour", r"pour (.+?) into (.+)", ["OBJ_from", "OBJ_to"])
    if action: return action
    action = _parse_general_two_arg_action(action_text_lower, "dunk", r"dunk (.+?) into (.+)", ["OBJ_container", "OBJ_liquid_env"])
    if action: return action

    # Device Operations
    action = _parse_general_one_arg_action(action_text_lower, "activate ", "OBJ")
    if action: return action
    action = _parse_general_one_arg_action(action_text_lower, "deactivate ", "OBJ")
    if action: return action
    
    # Special case: use {OBJ} [on {OBJ}]
    use_on_match = re.match(r"use (.+?) on (.+)", action_text_lower)
    if use_on_match:
        return {"tool_name": "use", "inputs": {"OBJ_tool": use_on_match.group(1).strip(), "OBJ_target": use_on_match.group(2).strip()}}
    
    # Must come after use_on_match to avoid misparsing
    action = _parse_general_one_arg_action(action_text_lower, "use ", "OBJ_tool")
    if action: return action


    # Miscellaneous
    action = _parse_general_one_arg_action(action_text_lower, "eat ", "OBJ_edible")
    if action: return action
    action = _parse_general_one_arg_action(action_text_lower, "flush ", "OBJ_flushable")
    if action: return action
    action = _parse_general_one_arg_action(action_text_lower, "focus on ", "OBJ") # OBJ is what to focus on
    if action: return action

    # wait [DURATION] (e.g., wait, wait1, wait 5)
    # Handles "wait", "waitN" (e.g. wait1), "wait N" (e.g. wait 5)
    # wait_match_simple = re.match(r"wait$", action_text_lower)
    # if wait_match_simple:
    #      return {"tool_name": "wait", "inputs": {"DURATION": 1}}

    # wait_match_num_suffix = re.match(r"wait(\d+)$", action_text_lower)
    # if wait_match_num_suffix:
    #     try:
    #         duration = int(wait_match_num_suffix.group(1))
    #         return {"tool_name": "wait", "inputs": {"DURATION": duration}}
    #     except ValueError:
    #         pass # Should not happen with \d+

    # wait_match_space_num = re.match(r"wait (\d+)$", action_text_lower)
    # if wait_match_space_num:
    #     try:
    #         duration = int(wait_match_space_num.group(1))
    #         return {"tool_name": "wait", "inputs": {"DURATION": duration}}
    #     except ValueError:
    #         pass


    # Fallback for unparsed actions
    if parsed_action["tool_name"] == "unknown":
        print(f"Warning: Action parsing resulted in 'unknown' for: '{action_text}'")
    return parsed_action

def infer_output_from_observation(action_parsed: Dict[str, Any], observation: str) -> Dict[str, Any]:
    """
    Infers structured output from ScienceWorld observation text.
    """
    output = {"raw_observation": observation, "status": "unknown"}
    tool_name = action_parsed.get("tool_name")
    # inputs = action_parsed.get("inputs", {})
    obs_lower = observation.lower()

    if "no known action matches that input" in obs_lower or \
       "it's not clear how to" in obs_lower or \
       "i'm not sure how to use those two things together" in obs_lower or \
       "unknown action" in obs_lower:
        output["status"] = "failure"
        output["error_message"] = observation
        return output

    if "you move to the" in obs_lower and tool_name == "go_to":
        output["status"] = "success"
        match = re.search(r"you move to the (.*?)\.", observation, re.IGNORECASE)
        if match:
            output["new_location"] = match.group(1)
    elif "you move the" in obs_lower and "to the inventory" in obs_lower and tool_name == "pick_up":
        output["status"] = "success"
        match = re.search(r"you move the (.*?) to the inventory", observation, re.IGNORECASE)
        if match:
            output["picked_object"] = match.group(1)
    elif "you move the" in obs_lower and tool_name == "put_down": # e.g. "You move the giant tortoise to the outside."
        output["status"] = "success"
        match = re.search(r"you move the (.*?) to the (.*?)\.", observation, re.IGNORECASE)
        if match:
            output["put_object"] = match.group(1)
            output["put_location"] = match.group(2) # usually the current location name
    elif tool_name == "look_around":
        output["status"] = "success"
        output["description_type"] = "location_details"
        # Room name
        room_match = re.search(r"This room is called the (.*?)\.|This outside location is called the (.*?)\.", observation, re.IGNORECASE)
        if room_match:
            output["current_room"] = room_match.group(1) or room_match.group(2)
        
        # Objects
        objects_seen = []
        # "In it, you see: \n\ta substance called air\n\ta painting"
        # "Here you see: \n\tthe agent\n\ta substance called air\n\tan axe"
        in_it_block_match = re.search(r"(?:In it, you see:|Here you see:)\s*\n(.*?)\n\s*(?:You also see:|A door to|$)", observation, re.DOTALL | re.IGNORECASE)
        if in_it_block_match:
            items_text = in_it_block_match.group(1)
            for line in items_text.split('\\n'): # Handle literal \\n from json
                 line = line.strip()
                 if line.startswith('\\t') or line.startswith('\t'):
                     item_desc = line.lstrip('\\t').lstrip('\t')
                     # Remove details like "(containing...)", "which is off"
                     simple_item = re.sub(r"\s*\(.*?\)|\s*, which is .*?$", "", item_desc).strip()
                     if simple_item and simple_item != "the agent":
                         objects_seen.append(simple_item)
        output["objects"] = list(set(objects_seen)) # Unique objects

        # Doors / Exits
        doors = []
        # "You also see:\n\tA door to the green house (that is open)"
        # Need to handle cases where there's no "You also see" but directly doors
        door_block_match = re.search(r"You also see:\s*\n(.*?)$", observation, re.DOTALL | re.IGNORECASE)
        door_lines_text = ""
        if door_block_match:
            door_lines_text = door_block_match.group(1)
        else: # If no "You also see:", check for doors directly from start of observation after room name
            if output.get("current_room"):
                room_name_end_idx = observation.lower().find(output["current_room"].lower()) + len(output["current_room"])
                potential_door_text = observation[room_name_end_idx:]
                if "door to" in potential_door_text.lower():
                    door_lines_text = potential_door_text

        if door_lines_text:
            for line in door_lines_text.split('\\n'): # Handle literal \\n from json
                line = line.strip()
                if (line.startswith('\\t') or line.startswith('\t')) and "door to" in line.lower():
                    door_desc = line.lstrip('\\t').lstrip('\t')
                    door_match = re.search(r"A door to the (.*?)\s*\((.*?)\)", door_desc, re.IGNORECASE)
                    if door_match:
                        doors.append({"name": f"door to the {door_match.group(1)}", "state": door_match.group(2)})
                    else: # simpler door format?
                        door_match_simple = re.search(r"A door to the (.*)", door_desc, re.IGNORECASE)
                        if door_match_simple:
                             doors.append({"name": f"door to the {door_match_simple.group(1)}", "state": "unknown"}) # if state not mentioned
        output["doors"] = doors
    elif tool_name == "look_at":
        output["status"] = "success"
        output["description_type"] = "object_details"
        # e.g. "a baby mouse", "a drawer. The drawer is closed."
        # Simplistic, assumes observation is the description
        output["object_description"] = observation.strip() 
        if "is closed" in obs_lower: output["state"] = "closed"
        elif "is open" in obs_lower: output["state"] = "open"
    elif tool_name == "look_in":
        output["status"] = "success"
        output["description_type"] = "container_contents"
        # "Inside the cupboard is: \n\ta ceramic cup (containing nothing)\n\ta drawer"
        # "Inside the fridge is: \n\ta wood cup (containing orange juice)"
        # "Inside the drawer is: \n\tnothing"
        if "nothing" in obs_lower and "inside" in obs_lower:
            output["contains"] = []
        else:
            items_inside = []
            inside_block_match = re.search(r"Inside the .*? is:\s*\n(.*?)$", observation, re.DOTALL | re.IGNORECASE)
            if inside_block_match:
                items_text = inside_block_match.group(1)
                for line in items_text.split('\\n'): # Handle literal \\n from json
                    line = line.strip()
                    if line.startswith('\\t') or line.startswith('\t'):
                        item_desc = line.lstrip('\\t').lstrip('\t')
                        simple_item = re.sub(r"\s*\(.*?\)|\s*, which is .*?$", "", item_desc).strip()
                        if simple_item:
                            items_inside.append(simple_item)
            output["contains"] = list(set(items_inside))
    elif tool_name in ["open", "close", "activate", "deactivate"]:
        # e.g. "The cupboard is now open." "The sink is now activated."
        if "is now open" in obs_lower or "is now closed" in obs_lower or \
           "is now activated" in obs_lower or "is now deactivated" in obs_lower:
            output["status"] = "success"
            obj_name_match = re.search(r"The (.*?) is now", observation, re.IGNORECASE)
            if obj_name_match:
                output["object_changed"] = obj_name_match.group(1)
            if "open" in obs_lower: output["new_state"] = "open"
            elif "closed" in obs_lower: output["new_state"] = "closed"
            elif "activated" in obs_lower: output["new_state"] = "activated"
            elif "deactivated" in obs_lower: output["new_state"] = "deactivated"
        elif "is already open" in obs_lower or "is already closed" in obs_lower:
            output["status"] = "success" # Technically success, no state change
            output["message"] = observation
    elif tool_name == "use" and "thermometer" in action_parsed.get("inputs", {}).get("OBJ_tool", "").lower():
        # "the thermometer measures a temperature of 13 degrees celsius"
        temp_match = re.search(r"the thermometer measures a temperature of ([\d\.]+) degrees celsius", obs_lower)
        if temp_match:
            output["status"] = "success"
            output["temperature_reading"] = float(temp_match.group(1))
            target_obj_match = re.search(r"on (.*?)$", action_parsed.get("inputs", {}).get("OBJ_target", ""), re.IGNORECASE) # try to get target from input if possible
            if target_obj_match:
                output["measured_object"] = target_obj_match.group(1).strip()
            elif "substance in" in action_parsed.get("inputs", {}).get("OBJ_target", ""):
                 output["measured_object"] = action_parsed.get("inputs", {}).get("OBJ_target", "")


    elif tool_name == "wait":
        if "you decide to wait for" in obs_lower: # ScienceWorld specific
            output["status"] = "success"
            match = re.search(r"for (\d+) iterations", obs_lower)
            if match:
                output["waited_iterations"] = int(match.group(1))
        elif "you wait for an iteration" in obs_lower: # General success for wait
             output["status"] = "success"
             output["waited_iterations"] = 1


    elif tool_name == "task":
        if "task description:" in obs_lower:
            output["status"] = "success"
            output["task_description"] = observation.split("Task description:\n")[-1].strip()
    elif tool_name == "inventory":
        output["status"] = "success"
        # "In your inventory, you see:\n\tan orange" or "Your inventory is empty."
        # Updated based on logs: "Your inventory contains: a thermometer..."
        if "your inventory is empty" in obs_lower or ("your inventory contains:" in obs_lower and "nothing" in obs_lower.split("your inventory contains:")[-1]):
            output["items"] = []
        else:
            items_inv = []
            # Match "In your inventory, you see:" or "Your inventory contains:"
            inv_block_match = re.search(r"(?:In your inventory, you see:|Your inventory contains:)\s*\n?(.*?)$", observation, re.DOTALL | re.IGNORECASE)
            if inv_block_match:
                items_text = inv_block_match.group(1).strip()
                if items_text.lower() == "nothing": # Double check for "nothing" after header
                     output["items"] = []
                else:
                    # Split by lines first, then parse items
                    for line in items_text.split('\\n'):
                        line = line.strip()
                        if line.startswith('\\t') or line.startswith('\t'):
                            item_desc = line.lstrip('\\t').lstrip('\t')
                            # Remove details like "(containing...)", "which is off"
                            simple_item = re.sub(r"\s*\(.*?\)|\s*, which is .*?$", "", item_desc).strip()
                            if simple_item:
                                items_inv.append(simple_item)
                        elif line and not (line.lower().startswith("in your inventory") or line.lower().startswith("your inventory contains")): # direct items without tab
                            simple_item = re.sub(r"\s*\(.*?\)|\s*, which is .*?$", "", line).strip()
                            if simple_item:
                                items_inv.append(simple_item)
                    # If no items were parsed by lines (e.g. comma separated on one line)
                    if not items_inv and items_text and not items_text.startswith("\\t") and not items_text.startswith("\t") and items_text.lower() != "nothing":
                         items_inv = [re.sub(r"\s*\(.*?\)|\s*, which is .*?$", "", i.strip()).strip() for i in re.split(r",\s*and\s*|,\s*|\s+and\s+", items_text) if i.strip()]

            output["items"] = list(set(items_inv)) # Unique items

    elif "you focus on the" in obs_lower and tool_name == "focus_on":
        output["status"] = "success"
        match = re.search(r"you focus on the (.*?)\.", observation, re.IGNORECASE)
        if match:
            output["focused_object"] = match.group(1)
    
    # If no specific handler matched but not an explicit failure message
    if output["status"] == "unknown" and "ambiguous request" not in obs_lower:
        # More general success/failure cues
        failure_keywords_general = [
            "you can't", "cannot", "is not possible", "nothing happens", 
            "is already open", "is already closed", "is not a container",
            "is not here", "is not in this room", "you do not see", "not in the inventory",
            "it is not possible to move it there", "you are not holding", "you are not carrying",
            "does not seem to have any effect", "is not an ingredient", "is not a device",
            "there is no"
        ]
        is_failure = any(keyword in obs_lower for keyword in failure_keywords_general)

        if is_failure:
            output["status"] = "failure"
            if not output.get("error_message"): output["error_message"] = observation
        else:
            # Broad success if no specific success/failure matched
            # This is a fallback and might need refinement based on ambiguous cases
            output["status"] = "success" # Default to success if not clearly failure

    return output

# Example Usage:
if __name__ == '__main__':
    actions_to_test = [
        "look around",
        "go to kitchen",
        "open cupboard",
        "pick up metal pot",
        "move metal pot to sink",
        "activate sink",
        "deactivate sink",
        "focus on substance in metal pot", # Multi-word object
        "pour metal pot into metal pot", # Will be parsed, but likely fail in env
        "move metal pot to stove",
        "activate stove",
        "use thermometer in inventory on substance in metal pot", # Complex use case
        "wait",
        "inventory",
        "task",
        "look at the red apple",
        "eat banana",
        "close fridge door" # Example where OBJ has multiple words
    ]

    print("--- Action Parsing Examples ---")
    for act_str in actions_to_test:
        parsed = parse_scienceworld_action(act_str)
        print(f"Action: '{act_str}' -> Parsed: {parsed}")

    print("\n--- Observation Parsing Examples ---")
    obs_tests = [
        ({"tool_name": "use", "inputs": {"OBJ_tool": "thermometer in inventory", "OBJ_target": "substance in metal pot"}}, 
         "The thermometer measures a temperature of 13 degrees celsius.", 
         "Success with temp"),
        ({"tool_name": "move", "inputs": {"OBJ_source": "metal pot", "OBJ_target": "metal pot"}}, 
         "You can't move something into itself.", 
         "Failure cant move"),
        ({"tool_name": "go_to", "inputs": {"LOC": "kitchen"}}, 
         "You move to the kitchen.", 
         "Success go to"),
        (parse_scienceworld_action("pick up metal pot"), 
         "You pick up the metal pot.", 
         "Success pick up"),
        (parse_scienceworld_action("pick up bottle"),
         "You move the bottle to the inventory.",
         "Success pick up (alt obs)"),
        (parse_scienceworld_action("open cupboard"),
         "You open the cupboard. The cupboard is now open. In the cupboard is: a tin cup (containing nothing), a ceramic cup (containing nothing), a drawer.",
         "Success open with content"),
        (parse_scienceworld_action("close fridge door"),
         "You close the fridge door. The fridge door is now closed.",
         "Success close"),
        (parse_scienceworld_action("inventory"),
         "Your inventory contains: a thermometer, a metal pot (containing a substance called water).",
         "Success inventory with items"),
        (parse_scienceworld_action("inventory"),
         "Your inventory contains: nothing.",
         "Success inventory empty"),
        (parse_scienceworld_action("look around"),
         "This room is called the kitchen. In it, you see: a substance called soap, a painting...",
         "Success look around"),
        (parse_scienceworld_action("wait"),
         "You wait for an iteration.",
         "Success wait default"),
        (parse_scienceworld_action("wait1"),
         "You wait for an iteration.",
         "Success wait1"),
        (parse_scienceworld_action("wait 5"),
         "You wait for 5 iterations.",
         "Success wait 5"),
         (parse_scienceworld_action("mix bowl"),
          "You mix the contents of the bowl. A new substance (pancake batter) is created.",
          "Success mix"),
         (parse_scienceworld_action("eat apple"),
          "You can't eat that. It's not edible.",
          "Failure eat non-edible")
    ]

    for parsed_act, obs_text, desc in obs_tests:
        if parsed_act: # Ensure action was parsed correctly for the test
            inferred_out = infer_output_from_observation(parsed_act, obs_text)
            print(f"Desc: {desc}\n  Action: {parsed_act}\n  Obs: '{obs_text}'\n  Inferred: {inferred_out}\n")
        else:
            print(f"Skipping observation test for '{desc}' due to action parsing issue.")