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
from autool.utils.parser.alfworld import parse_alfworld_action, infer_output_from_observation 

class AlfworldAdaptor(EnvironmentAdapter):
    def __init__(self, debug: bool = False):
        """
        初始化环境适配器
        Args:
            debug: 是否启用调试模式
        """
        self.debug = debug
        self.goal = None
        self.state = {
            "objects": set(),
            "receptacles": set(),     # 场景中的容器
            "object_locations": {},   # 对象位置 {物体: 位置/'agent'}
            "receptacle_states": defaultdict(lambda: 'unknown'), # 容器状态 {容器名: "open"/"closed"/"unknown"}
            "current_location": None,    # 当前位置
            "inventory": set(),        # 物品栏中的物品 (通过 object_locations 推断)
            "last_observations": defaultdict(lambda: None) # 上一个观察
        }

    def reset(self, init_observation: str = None):
        self.goal = None
        self.state = {
            "objects": set(),
            "receptacles": set(),     # 场景中的容器
            "object_locations": {},   # 对象位置 {物体: 位置/'agent'}
            "receptacle_states": defaultdict(lambda: 'unknown'), # 容器状态 {容器名: "open"/"closed"/"unknown"}
            "current_location": None,    # 当前位置
            "inventory": set(),        # 物品栏中的物品 (通过 object_locations 推断)
            "last_observations": defaultdict(lambda: None) # 上一个观察
        }
        pass

    def parse_action(self, action_text: str, tool_descriptions: Dict) -> Optional[Dict[str, Any]]:
        parsed = parse_alfworld_action(action_text, tool_descriptions)
        return parsed

    def infer_output(self, tool_name: str, inputs: Dict[str, Any], result: Any) -> Dict[str, Any]:
        parsed = infer_output_from_observation({"tool_name": tool_name, "inputs": inputs}, result)
        return parsed

    def update_state(self, action_parsed: Dict[str, Any], structured_outputs: Dict[str, Any]) -> None:
        try:
            tool_name = action_parsed.get("tool_name")
            if not tool_name:
                return
            self.state["last_observations"][tool_name] = structured_outputs
            if tool_name == "go" and structured_outputs.get("status") == "success":
                loc = action_parsed.get("inputs", {}).get("recep")
                if loc:
                    self.state["current_location"] = loc
            elif tool_name == "take" and structured_outputs.get("status") == "success":
                obj = action_parsed.get("inputs", {}).get("obj")
                if obj: 
                    self.state["inventory"].add(obj)
                    self.state["object_locations"][obj] = "agent"
            elif tool_name == "put" and structured_outputs.get("status") == "success":
                obj = action_parsed.get("inputs", {}).get("obj")
                loc = action_parsed.get("inputs", {}).get("recep")
                if obj and loc: 
                    self.state["object_locations"][obj] = loc
                    self.state["inventory"].discard(obj)
            elif tool_name == "examine" and structured_outputs.get("status") == "success":
                # 处理 examine 动作的输出
                examined_content = structured_outputs.get("examined_content")
                if examined_content:
                    for item in examined_content:
                        self.state["objects"].add(item)
            elif tool_name == "open" and structured_outputs.get("status") == "success":
                receptacle = action_parsed.get("inputs", {}).get("recep")
                if receptacle: 
                    self.state["receptacle_states"][receptacle] = "open"
            elif tool_name == "close" and structured_outputs.get("status") == "success":
                receptacle = action_parsed.get("inputs", {}).get("recep")
                if receptacle:
                    self.state["receptacle_states"][receptacle] = "closed"

        except Exception as e:
            if self.debug:
                print(f"更新状态时出错: {e}")

    def get_contextual_params(self, action_type: str, missing_params: Set[str], required_params_info: Dict) -> Dict[str, Any]:
        """
        根据当前维护的环境状态推断参数值
        
        Args:
            action_type: 目标工具/动作类型
            missing_params: 需要填充的参数集合
            required_params_info: 参数的详细信息字典
            
        Returns:
            Dict[str, Any]: 从环境上下文中推断出的参数值字典
        """
        filled = {}
        if self.debug: print(f"  Phase 3: Checking Environment State for {missing_params}...")

        # 提取目标中的物体和位置
        goal_objects = set()
        goal_locations = set()
        if hasattr(self, 'goal') and self.goal:
            goal_lower = self.goal.lower()

            # 查找已知对象/容器名称是否在目标中
            for obj in self.state.get("objects", set()).union(self.state.get("receptacles", set())):
                if obj in goal_lower:
                    if obj in self.state.get("receptacles", set()): goal_locations.add(obj)
                    else: goal_objects.add(obj)

            # 尝试提取 'X in/on Y' 结构
            put_match = re.search(r"put (?:a|an|the) (.*?) (?:in|on) (?:a|an|the) (.*?)\.", goal_lower)
            if put_match:
                goal_objects.add(put_match.group(1).strip())
                goal_locations.add(put_match.group(2).strip())
                
            # 尝试提取 'look at X under/in/on Y'
            look_match = re.search(r"look at (?:a|an|the) (.*?) (?:under|in|on) (?:a|an|the) (.*?)\.", goal_lower)
            if look_match:
                goal_objects.add(look_match.group(1).strip())
                goal_locations.add(look_match.group(2).strip())

        if self.debug: 
            print(f"    Goal Info: Objects={goal_objects}, Locations={goal_locations}")
            print(f"    Current State: Loc={self.state.get('current_location')}, Inv={self.state.get('inventory')}, " 
                f"ObjLocs={self.state.get('object_locations')}, RecepStates={self.state.get('receptacle_states')}")

        params_to_try = list(missing_params)
        for param_name in params_to_try:
            param_info = required_params_info.get(param_name, {})
            param_type = param_info.get("type", "str")
            value = None
            normalized_param = param_name.lower()

            # --- 推断逻辑 ---

            # 1. 推断对象参数 ('obj', 'target' for non-receptacles)
            if any(p in normalized_param for p in ["obj", "item", "thing"]) or \
            (normalized_param == "target" and action_type not in ["go", "examine", "open", "close"]):

                candidates = set()
                # 优先级1: 目标中提到的对象
                candidates.update(goal_objects)

                current_location = self.state.get("current_location")
                inventory = self.state.get("inventory", set())
                object_locations = self.state.get("object_locations", {})
                receptacles = self.state.get("receptacles", set())

                if action_type == "put": # 必须在物品栏
                    candidates = candidates.intersection(inventory)
                    if not candidates: candidates = inventory # Fallback: any inventory item
                elif action_type == "take":
                    # 应该在当前位置，且不在物品栏
                    if current_location:
                        candidates_at_loc = {obj for obj, obj_loc in object_locations.items() if obj_loc == current_location and obj not in receptacles}
                        candidates = candidates.intersection(candidates_at_loc)
                        if not candidates: candidates = candidates_at_loc - inventory
                    else:
                        # 如果有recently_seen_objects属性，使用它
                        recently_seen = getattr(self, 'recently_seen_objects', set())
                        candidates = {o for o in recently_seen if o not in inventory and o not in receptacles}

                elif action_type in ["clean", "cool", "heat", "slice", "use"]:
                    # 优先级: 物品栏中的目标物品 > 当前位置的目标物品 > 物品栏其他 > 当前位置其他
                    inv_goal_obj = candidates.intersection(inventory)
                    loc_goal_obj = candidates.intersection({o for o, l in object_locations.items() if l == current_location})
                    other_inv = inventory - candidates
                    other_loc = {o for o, l in object_locations.items() if l == current_location} - candidates - inventory
                    objects = self.state.get("objects", set())

                    if inv_goal_obj: candidates = inv_goal_obj
                    elif loc_goal_obj: candidates = loc_goal_obj
                    elif other_inv: candidates = other_inv
                    elif other_loc: candidates = other_loc
                    else: candidates = objects - receptacles

                # 选择一个候选者 (随机或第一个)
                if candidates: value = random.choice(list(candidates))

            # 2. 推断位置/容器参数 ('recep', 'destination', 'source', 'location', 'target' for containers)
            elif any(p in normalized_param for p in ["recep", "destination", "source", "location"]) or \
                (normalized_param == "target" and action_type in ["go", "examine", "open", "close"]):

                candidates = set()
                # 优先级1: 目标中提到的位置
                candidates.update(goal_locations)

                current_location = self.state.get("current_location")
                inventory = self.state.get("inventory", set())
                object_locations = self.state.get("object_locations", {})
                receptacles = self.state.get("receptacles", set())
                receptacle_states = self.state.get("receptacle_states", defaultdict(lambda: 'unknown'))

                if action_type == "go":
                    # 优先去目标位置，其次去包含目标物品的位置，然后去未知/未探索位置
                    target_obj_loc = set()
                    for obj in goal_objects:
                        if obj in object_locations and object_locations[obj] != 'agent':
                            target_obj_loc.add(object_locations[obj])

                    unknown_state_receps = {r for r, s in receptacle_states.items() if s == 'unknown'}
                    other_receps = receptacles - candidates - target_obj_loc - unknown_state_receps

                    if candidates: pass # Already has goal locations
                    elif target_obj_loc: candidates = target_obj_loc
                    elif unknown_state_receps: candidates = unknown_state_receps
                    else: candidates = other_receps

                elif action_type == "take": # Source receptacle
                    # 应该是当前位置
                    if current_location and current_location in receptacles:
                        candidates = {current_location}
                    # Fallback: 目标物品的位置（如果不在当前位置但已知）
                    elif goal_objects:
                        obj_to_take = next(iter(goal_objects), None)
                        if obj_to_take and obj_to_take in object_locations and object_locations[obj_to_take] != 'agent':
                            candidates = {object_locations[obj_to_take]}

                elif action_type == "put": # Destination receptacle
                    # 优先去目标位置，其次去当前位置（如果合适），然后去开放容器
                    open_receps = {r for r, s in receptacle_states.items() if s == 'open'}
                    current_loc_set = {current_location} if current_location in receptacles else set()

                    if candidates: pass # Goal location first
                    # 优先选择开放的目标容器或当前位置的开放容器
                    open_goal_receps = candidates.intersection(open_receps)
                    open_current_loc_receps = current_loc_set.intersection(open_receps)

                    if open_goal_receps: candidates = open_goal_receps
                    elif open_current_loc_receps: candidates = open_current_loc_receps
                    elif current_loc_set: candidates = current_loc_set
                    elif open_receps - candidates: candidates = open_receps - candidates
                    else: candidates = receptacles

                elif action_type in ["examine", "open", "close"]: # Target receptacle
                    # 优先当前位置，其次目标位置，然后根据状态选择
                    current_loc_set = {current_location} if current_location in receptacles else set()
                    if current_loc_set: candidates = current_loc_set
                    elif candidates: pass # Goal location
                    else:
                        if action_type == "open": candidates = {r for r, s in receptacle_states.items() if s != 'open'} or receptacles
                        elif action_type == "close": candidates = {r for r, s in receptacle_states.items() if s == 'open'} or receptacles
                        else: candidates = receptacles # Examine any

                elif action_type in ["clean", "cool", "heat"]: # Tool receptacle
                    # 需要常识或规则
                    if action_type == "clean": candidates = {r for r in receptacles if "sink" in r}
                    elif action_type == "cool": candidates = {r for r in receptacles if "fridge" in r}
                    elif action_type == "heat": candidates = {r for r in receptacles if "microwave" in r or "stove" in r}
                    if not candidates: candidates = receptacles # Fallback

                # 选择一个候选者
                if candidates: value = random.choice(list(candidates))

            # --- 赋值 ---
            if value is not None:
                if self._check_type(value, param_type):
                    filled[param_name] = value
                    missing_params.remove(param_name)
                    if self.debug: print(f"      Inferred '{param_name}' from env state = {value}")

        return filled

    def generate_action_from_params(self, action_type: str, params: Dict[str, Any]) -> str:
        """根据参数生成完整的动作"""
        if action_type == "go":
            destination = params.get("recep", "")
            if destination:
                return f"go to {destination}"
        
        elif action_type == "take":
            obj = params.get("obj", "")
            source = params.get("recep", "")
            if obj and source:
                return f"take {obj} from {source}"
        
        elif action_type == "put":
            obj = params.get("obj", "")
            recep = params.get("recep", "")
            if obj and recep:
                return f"put {obj} in/on {recep}"
            elif obj:
                return f"put {obj}"
        
        elif action_type == "examine":
            target = params.get("target", "")
            if target:
                return f"examine {target}"
        
        elif action_type == "clean" or action_type == "heat" or action_type == "cool":
            obj = params.get("obj", "")
            recep = params.get("recep", "")
            if obj and recep:
                return f"clean {obj} with {recep}"
        
        elif action_type == "check":
            return f"check valid actions"
        
        # 对于其他动作类型，直接拼接参数
        param_values = list(params.values())
        if param_values:
            return f"{action_type} {' '.join(str(v) for v in param_values)}"
        
        # 兜底返回
        return action_type


class AlfworldParamCompletion:
    """
    适用于学术/电影/天气等数据集的参数填充类，使用新的框架实现
    """
    
    def __init__(self,
        tool_graph: ToolGraph,
        tool_descriptions: Dict[str, Dict],
        dependency_graph_path: Optional[str] = None,
        history_max_len: int = 10,
        goal: Optional[str] = None,
        debug: bool = False):
        """
        初始化参数填充器
        Args:
            tool_graph: 工具调用图
            tool_descriptions: 工具描述字典
            dependency_graph_path: 参数依赖图文件路径
            history_max_len: 执行历史的最大长度
            goal: 当前目标
            debug: 是否启用调试模式
        """
        self.debug = debug
        
        # 创建适配器
        self.adapter = AlfworldAdaptor(debug=debug)
        
        # 创建框架
        self.framework = ParameterFillingFramework(
            tool_graph=tool_graph,
            tool_descriptions=tool_descriptions,
            adapter=self.adapter,
            param_dependency_path=dependency_graph_path,
            history_max_len=history_max_len,
            debug=debug
        )
            
    def record_execution(self, action_text: str, result: Any):
        """
        记录工具执行
        Args:
            action_text: 动作文本
            result: 执行结果
        """
        self.framework.record_execution(action_text=action_text, action_parsed=parse_alfworld_action(action_text), result=result)
    
    def fill_parameters(self, target_tool: str, existing_params: Optional[Dict[str, Any]] = None, lookback_k: int = 5) -> Dict[str, Any]:
        """
        填充参数
        Args:
            target_tool: 目标工具名称
            existing_params: 已有的参数字典
            lookback_k: 历史回溯步数
        Returns:
            填充后的参数字典
        """
        return self.framework.fill_parameters(
            target_tool=target_tool,
            existing_params=existing_params,
            lookback_k=lookback_k
        )
    
    def set_debug(self, debug: bool):
        """
        设置调试模式
        Args:
            debug: 是否启用调试模式
        """
        self.debug = debug
        self.framework.set_debug(debug)

if __name__ == '__main__':

    pass