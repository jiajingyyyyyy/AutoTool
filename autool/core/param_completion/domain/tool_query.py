from autool.core.tool_predict.datastruct import ToolGraph
from typing import Dict, List, Any, Tuple, Optional, Set
import re
import os
import random # 用于随机选择
from ..param_dependency import ExecutionHistory
from collections import defaultdict
import json     
from ..param_completion import EnvironmentAdapter, ParameterFillingFramework
from autool.utils.parser.tool_query import parse_tool_query

class GenericDatasetAdapter(EnvironmentAdapter):
    """
    学术/电影/天气等数据集的通用适配器实现
    """
    
    def __init__(self, debug: bool = False):
        """
        初始化通用数据集适配器
        Args:
            debug: 是否启用调试模式
        """
        super().__init__(debug)
        
        # 状态跟踪
        self.current_graphs_loaded: Set[str] = set()       # 当前加载的图（例如'paper', 'author'）
        self.last_checked_node: Optional[Dict] = None      # 最近检查的节点
        self.last_searched_entity: Optional[Dict] = None   # 最近搜索的实体
        self.last_location_queried: Optional[Dict] = None  # 最近查询的位置
        self.current_task_context: Dict[str, Any] = {}     # 当前任务上下文
        self.goal: Optional[str] = None      
                      # 当前目标
    def reset(self, init_observation: str = None):
        """
        重置适配器状态，可选地传入初始observation。
        Args:
            init_observation: 初始环境观测（可选）
        """
        self.current_graphs_loaded = set()
        self.last_checked_node = None
        self.last_searched_entity = None
        self.last_location_queried = None
        self.current_task_context = {}
        self.goal = None
        if init_observation:
            # 可根据需要解析初始observation，更新状态
            # 例如：提取初始节点、图等
            if self.debug:
                print(f"Adapter reset with initial observation: {init_observation}")
        if self.debug:
            print("Adapter state has been reset.")


    def parse_action(self, action_text: str, tool_descriptions: Dict) -> Optional[Dict[str, Any]]:
        parsed = parse_tool_query(action_text, tool_descriptions)
        return parsed
    
    # def infer_output(self, tool_name: str, inputs: Dict[str, Any], result: Any) -> Dict[str, Any]:
    #     """
    #     根据工具执行结果推断结构化输出
    #     Args:
    #         tool_name: 工具名称
    #         inputs: 输入参数
    #         result: 执行结果
    #     Returns:
    #         推断的结构化输出字典
    #     """
    #     outputs = {"status": "unknown"}
        
    #     # 检查错误状态
    #     if isinstance(result, (str, bytes)) and ("Error" in str(result) or "Failed" in str(result)):
    #         outputs["status"] = "failure"
    #         outputs["error_message"] = str(result)
    #     elif result is not None: # 假设成功
    #         outputs["status"] = "success"
            
    #         # 根据工具类型处理结果
    #         if tool_name == "loadPaperNet":
    #             outputs["loaded_graph"] = "paper"
    #         elif tool_name == "loadAuthorNet":
    #             outputs["loaded_graph"] = "author"
    #         elif tool_name == "neighbourCheck":
    #             if isinstance(result, list):
    #                 outputs["neighbors"] = result
    #             outputs["node_id"] = inputs.get("node")
    #             outputs["node_type"] = inputs.get("graph")
    #         elif tool_name == "search_movie":
    #             if isinstance(result, dict):
    #                 outputs.update(result) # 假设结果是字典，如 {"movie_id": ..}
    #         elif tool_name == "search_person":
    #             if isinstance(result, dict):
    #                 outputs.update(result)
    #         elif tool_name == "get_latitude_longitude":
    #             if isinstance(result, dict):
    #                 outputs.update(result)
    #                 outputs["location"] = inputs.get("location")
    #         elif tool_name == "get_forecast":
    #             if isinstance(result, dict):
    #                 outputs["forecast_data"] = result
    #             outputs["latitude"] = inputs.get("latitude")
    #             outputs["longitude"] = inputs.get("longitude")
    #         # 其他工具的通用处理
    #         else:
    #             # 简单类型直接存储
    #             if isinstance(result, (str, int, float, bool, list)):
    #                 outputs["result_value"] = result
    #             # 字典类型合并
    #             elif isinstance(result, dict):
    #                 outputs.update(result)
    #     else:
    #         outputs["status"] = "failure_no_result"
            
    #     return outputs
    def infer_output(self, tool_name: str, inputs: Dict[str, Any], observation: str) -> Dict[str, Any]:
        """
        根据工具执行结果（observation字符串）推断结构化输出 (优化版本 for tool-query)
        Args:
            tool_name: 工具名称 (小写化以便匹配)
            inputs: 输入参数 (可用于辅助解析，例如知道查询的节点或图)
            observation: 原始观察字符串
        Returns:
            推断的结构化输出字典
        """
        outputs: Dict[str, Any] = {"raw_observation": observation}
        normalized_tool_name = tool_name.lower().replace(" ", "") # 规范化工具名

        # 1. 检查通用错误模式 (Tool-query 环境的常见错误提示)
        obs_lower = observation.lower()
        if "invalid action" in obs_lower or "format error" in obs_lower or "error" in obs_lower:
            outputs["status"] = "failure"
            outputs["error_message"] = observation
            if self.debug: print(f"[InferOutput] Detected error for {tool_name}: {observation}")
            return outputs

        # 2. 根据规范化的工具名称进行精确解析 (Tool-query Task Tools)
        outputs["status"] = "success" # 假设没有检测到错误就是成功

        if normalized_tool_name == "loadpapernet":
            if "papernet is loaded" in obs_lower:
                outputs["loaded_graph_name"] = "PaperNet"
                if self.debug: print(f"[InferOutput] {tool_name}: Parsed loaded graph name: PaperNet")

        elif normalized_tool_name == "loadauthornet":
            if "authornet is loaded" in obs_lower:
                outputs["loaded_graph_name"] = "AuthorNet"
                if self.debug: print(f"[InferOutput] {tool_name}: Parsed loaded graph name: AuthorNet")

        # *** 重点修改这里和下面的列表/字典解析部分 ***
        elif normalized_tool_name in ["neighbourcheck", "neighborcheck", "authoredgecheck", "paperedgecheck"]:
            observation_stripped = observation.strip()
            if observation_stripped.startswith('[') and observation_stripped.endswith(']'):
                try:
                    json_compatible_obs = observation_stripped.replace("'", '"')
                    entities_list = json.loads(json_compatible_obs)
                    outputs["entities_list"] = entities_list
                    outputs["entity_count"] = len(entities_list)
                    if self.debug: print(f"[InferOutput] {tool_name}: Parsed entity list (count: {len(entities_list)})")
                except (json.JSONDecodeError, TypeError) as e:
                    outputs["status"] = "failure_parse"
                    outputs["parse_error"] = f"Failed to parse list: {e}"
                    if self.debug: print(f"[InferOutput] {tool_name}: Failed to parse list: {observation} - {e}")
            elif observation_stripped.startswith('{') and observation_stripped.endswith('}'):
                try:
                    json_compatible_obs = observation_stripped.replace("'", '"')
                    dict_result = json.loads(json_compatible_obs)
                    # 直接将字典的每个 key-value 合并到 outputs
                    outputs.update(dict_result)
                    if self.debug: print(f"[InferOutput] {tool_name}: Parsed dict and merged to outputs: {dict_result}")
                except (json.JSONDecodeError, TypeError) as e:
                    outputs["status"] = "failure_parse"
                    outputs["parse_error"] = f"Failed to parse dict: {e}"


            elif "there is no" in obs_lower:
                outputs["entities_list"] = []
                outputs["entity_count"] = 0
                outputs["result_message"] = observation
                if self.debug: print(f"[InferOutput] {tool_name}: Parsed empty entity list (not found)")
            else:
                outputs["status"] = "failure_format"
                outputs["parse_error"] = "Unexpected observation format for entity check"
                if self.debug: print(f"[InferOutput] {tool_name}: Unexpected format: {observation}")

        elif normalized_tool_name in ["authornodecheck", "papernodecheck"]:
            observation_stripped = observation.strip()
            if observation_stripped.startswith('{') and observation_stripped.endswith('}'):
                try:
                    # *** FIX: Ensure single quotes are handled - Already done in previous version, but double check ***
                    json_compatible_obs = observation_stripped.replace("'", '"')
                    node_info = json.loads(json_compatible_obs)
                    outputs["node_details"] = node_info
                    outputs["found"] = True
                    if 'name' in node_info: outputs['node_name'] = node_info['name']
                    if 'id' in node_info: outputs['node_id'] = node_info['id']
                    if self.debug: print(f"[InferOutput] {tool_name}: Parsed node details: {node_info}")
                except (json.JSONDecodeError, TypeError) as e:
                    outputs["status"] = "failure_parse"
                    outputs["parse_error"] = f"Failed to parse dict: {e}"
                    if self.debug: print(f"[InferOutput] {tool_name}: Failed to parse dict: {observation} - {e}")

            elif "there is no node" in obs_lower:
                outputs["node_details"] = None
                outputs["found"] = False
                outputs["result_message"] = observation
                if self.debug: print(f"[InferOutput] {tool_name}: Parsed node not found")
            else:
                outputs["status"] = "failure_format"
                outputs["parse_error"] = "Unexpected observation format for node check"
                if self.debug: print(f"[InferOutput] {tool_name}: Unexpected format: {observation}")

        elif normalized_tool_name == "check_valid_actions":
             # 观察示例: "You can use following valid actions: ['tool1', 'tool2', ...]"
             match = re.search(r"\[(.*?)\]", observation)
             if match:
                 try:
                     # *** FIX: Replace single quotes with double quotes for JSON compatibility ***
                    json_compatible_list_str = "[" + match.group(1).replace("'", '"') + "]"
                    valid_actions_list = json.loads(json_compatible_list_str)
                    outputs["valid_actions_list"] = valid_actions_list
                    if self.debug: print(f"[InferOutput] {tool_name}: Parsed valid actions list (count: {len(valid_actions_list)})")
                 except (json.JSONDecodeError, TypeError) as e:
                     outputs["status"] = "failure_parse"
                     outputs["parse_error"] = f"Failed to parse list: {e}"
                     if self.debug: print(f"[InferOutput] {tool_name}: Failed to parse list: {observation} - {e}")
             else:
                 outputs["status"] = "failure_format"
                 outputs["parse_error"] = "Valid actions list format not found in observation"
                 if self.debug: print(f"[InferOutput] {tool_name}: Unexpected format: {observation}")


        elif normalized_tool_name == "finish":
             outputs["final_result_value"] = observation
             if self.debug: print(f"[InferOutput] {tool_name}: Parsed final result: {observation}")

        # 3. 其他未精确处理的工具的通用处理 (仅作为回退)
        # ... (可以保留原有的通用逻辑，但确保不会覆盖前面精确解析的状态) ...
        # 例如，如果 status 仍是 'success' 且没有特定的输出参数被添加，
        # 可以考虑添加一个通用的输出参数。

        # 示例：如果上面没有解析出任何特定的结构化输出（outputs字典大小没变），可以考虑将整个观察作为通用输出
        # if len(outputs) == 1 and "raw_observation" in outputs: # 如果只有 raw_observation
        #      outputs["generic_observation_result"] = observation


        if self.debug: print(f"[InferOutput] Final outputs for {tool_name}: {outputs}")
        return outputs
    
    def generate_action_from_params(self, action_type: str, params: Dict[str, Any]) -> str:
        """
        根据参数生成标准的Action字符串，符合tool_query_sys_prompt.py的格式。
        Args:
            action_type: 工具/动作名称
            params: 参数字典
        Returns:
            标准格式的动作字符串
        """
        # 保证参数顺序与输入一致
        params_str = json.dumps(params, ensure_ascii=False, separators=(", ", ": "))
        return f"{action_type} with Action Input: {params_str}"

    def update_state(self, action_parsed: Dict[str, Any], structured_outputs: Dict[str, Any]) -> None:
        """
        根据动作和输出更新环境状态
        Args:
            action_parsed: 解析后的动作字典
            structured_outputs: 结构化的输出字典
        """
        tool_name = action_parsed.get("tool_name", "")
        inputs = action_parsed.get("inputs", {})
        status = structured_outputs.get("status")
        
        # 只在成功时更新状态
        if status == "success":
            # 更新图加载状态
            if tool_name == "loadPaperNet":
                self.current_graphs_loaded.add("paper")
            elif tool_name == "loadAuthorNet":
                self.current_graphs_loaded.add("author")
            # 更新节点检查状态
            elif tool_name == "neighbourCheck":
                self.last_checked_node = {
                    "type": inputs.get("graph"),
                    "id": inputs.get("node"),
                    "neighbors": structured_outputs.get("neighbors", [])
                }
            elif tool_name == "authorNodeCheck":
                self.last_checked_node = {
                    "type": "author",
                    "id": inputs.get("node"),
                    "details": structured_outputs
                }
            elif tool_name == "paperNodeCheck":
                self.last_checked_node = {
                    "type": "paper",
                    "id": inputs.get("node"),
                    "details": structured_outputs
                }
            # 更新实体搜索状态
            elif tool_name == "search_movie":
                self.last_searched_entity = {
                    "type": "movie",
                    "id": structured_outputs.get("movie_id"),
                    "name": inputs.get("name")
                }
            elif tool_name == "search_person":
                self.last_searched_entity = {
                    "type": "person",
                    "id": structured_outputs.get("person_id"),
                    "name": inputs.get("name")
                }
            # 更新位置查询状态
            elif tool_name == "get_latitude_longitude":
                self.last_location_queried = {
                    "location": inputs.get("location"),
                    "latitude": structured_outputs.get("latitude"),
                    "longitude": structured_outputs.get("longitude")
                }
        
        if self.debug:
            print(f"  State Updated: Graphs={self.current_graphs_loaded}, LastNode={self.last_checked_node}, LastEntity={self.last_searched_entity}, LastLoc={self.last_location_queried}")
    
    def get_contextual_params(self, action_type: str, missing_params: Set[str], required_params_info: Dict) -> Dict[str, Any]:
        """
        根据当前环境状态推断参数值
        Args:
            action_type: 目标动作类型
            missing_params: 缺失的参数集合
            required_params_info: 必需的参数描述字典
        Returns:
            推断的参数字典
        """
        filled = {}
        if self.debug:
            print(f"  Getting contextual params for {action_type}, missing: {missing_params}")
            print(f"  Context: Graphs={self.current_graphs_loaded}, LastNode={self.last_checked_node}, LastEntity={self.last_searched_entity}, LastLoc={self.last_location_queried}, TaskCtx={self.current_task_context}")
        
        params_to_try = list(missing_params)
        for param_name in params_to_try:
            param_info = required_params_info.get(param_name, {})
            param_type = param_info.get("type", "str")
            value = None
            normalized_param = param_name.lower()
            
            # --- 推断逻辑 ---
            
            # 1. 图类型 ('graph')
            if normalized_param == 'graph' and action_type in ['neighbourCheck', 'neighborCheck']:
                # 优先使用与上次节点相关的图
                if self.last_checked_node and self.last_checked_node.get('type') in self.current_graphs_loaded:
                    value = self.last_checked_node['type']
                # 尝试从实体类型推断图类型
                elif self.last_searched_entity:
                    entity_type = self.last_searched_entity.get('type')
                    if entity_type == 'person' and 'author' in self.current_graphs_loaded:
                        value = 'author'
                # 从任务上下文获取
                elif 'target_graph' in self.current_task_context and self.current_task_context['target_graph'] in self.current_graphs_loaded:
                    value = self.current_task_context['target_graph']
                # 使用任意已加载的图
                elif self.current_graphs_loaded:
                    value = next(iter(self.current_graphs_loaded))
            
            # 2. 节点 ID ('node', 'node1', 'node2')
            elif normalized_param.startswith('node'):
                # 优先使用上次检查的节点
                if self.last_checked_node and self._check_node_type_match(self.last_checked_node, action_type):
                    value = self.last_checked_node.get('id')
                # 使用上次搜索的实体ID
                elif self.last_searched_entity and self._check_entity_type_match(self.last_searched_entity, action_type):
                    value = self.last_searched_entity.get('id')
                # 从任务上下文获取
                elif 'target_node' in self.current_task_context:
                    value = self.current_task_context['target_node']
                elif 'target_author_id' in self.current_task_context and action_type.startswith('author'):
                    value = self.current_task_context['target_author_id']
                elif 'target_paper_id' in self.current_task_context and action_type.startswith('paper'):
                    value = self.current_task_context['target_paper_id']
                # 特殊处理node2参数
                elif normalized_param == 'node2' and self.last_checked_node and self.last_checked_node.get('neighbors'):
                    value = random.choice(self.last_checked_node['neighbors'])
            
            # 3. 实体 ID ('id' for movies/persons)
            elif normalized_param == 'id':
                if action_type == 'get_featured_movies' and self.last_searched_entity and self.last_searched_entity.get('type') == 'person':
                    value = self.last_searched_entity.get('id')
                elif action_type == 'get_featured_movies' and 'target_person_id' in self.current_task_context:
                    value = self.current_task_context['target_person_id']
                elif action_type == 'get_movie_credits' and self.last_searched_entity and self.last_searched_entity.get('type') == 'movie':
                    value = self.last_searched_entity.get('id')
                elif action_type == 'get_movie_credits' and 'target_movie_id' in self.current_task_context:
                    value = self.current_task_context['target_movie_id']
            
            # 4. 名称 ('name' for search)
            elif normalized_param == 'name':
                if action_type == 'search_person':
                    value = self.current_task_context.get('target_person_name') or self.current_task_context.get('target_author')
                elif action_type == 'search_movie':
                    value = self.current_task_context.get('target_movie_name')
            
            # 5. 地理位置 ('location', 'latitude', 'longitude')
            elif normalized_param == 'location' and action_type == 'get_latitude_longitude':
                value = self.current_task_context.get('target_city') or self.current_task_context.get('target_location')
            elif normalized_param == 'latitude' and self.last_location_queried:
                value = self.last_location_queried.get('latitude')
            elif normalized_param == 'longitude' and self.last_location_queried:
                value = self.last_location_queried.get('longitude')
            
            # 6. 答案 ('answer' for finish)
            elif normalized_param == 'answer' and action_type == 'finish':
                pass  # 这通常需要LLM来填充，不太适合自动推断
            
            # --- 赋值 ---
            if value is not None:
                if self._check_type(value, param_type):
                    filled[param_name] = value
                    missing_params.remove(param_name)
                    if self.debug:
                        print(f"      Inferred '{param_name}' from context state = {value}")
        
        return filled
    
    def set_goal(self, goal: str) -> None:
        """
        设置当前目标
        Args:
            goal: 目标文本
        """
        self.goal = goal
        if self.debug:
            print(f"Goal set to: {self.goal}")
        # 解析目标更新上下文
        self._parse_goal_for_context()
    
    def _parse_goal_for_context(self) -> None:
        """
        从目标中解析上下文信息
        """
        self.current_task_context = {}
        if not self.goal:
            return
            
        goal_lower = self.goal.lower()
        
        # 提取城市名称（用于天气查询）
        city_match = re.search(r"weather(?:\s+condition)?(?:\s+in|of|for|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", self.goal, re.IGNORECASE)
        if city_match:
            self.current_task_context['target_city'] = city_match.group(1)
            self.current_task_context['target_location'] = city_match.group(1)
        
        # 提取学术/论文相关实体
        if "author" in goal_lower:
            author_match = re.search(r"author\s+(?:named\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", self.goal, re.IGNORECASE)
            if author_match:
                self.current_task_context['target_author'] = author_match.group(1)
        
        if "paper" in goal_lower:
            paper_match = re.search(r"paper\s+(?:titled\s+)?[\"']([^\"']+)[\"']", self.goal)
            if paper_match:
                self.current_task_context['target_paper_title'] = paper_match.group(1)
        
        # 提取电影/人物相关实体
        if "movie" in goal_lower:
            movie_match = re.search(r"movie\s+(?:titled\s+)?[\"']([^\"']+)[\"']", self.goal)
            if movie_match:
                self.current_task_context['target_movie_name'] = movie_match.group(1)
        
        if "person" in goal_lower or "actor" in goal_lower or "actress" in goal_lower or "director" in goal_lower:
            person_match = re.search(r"(?:person|actor|actress|director)\s+(?:named\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", self.goal, re.IGNORECASE)
            if person_match:
                self.current_task_context['target_person_name'] = person_match.group(1)
        
        if self.debug:
            print(f"  Parsed Context from Goal: {self.current_task_context}")
    
    def _check_node_type_match(self, node_info: Dict, action_type: str) -> bool:
        """
        检查节点类型是否与动作类型匹配
        Args:
            node_info: 节点信息
            action_type: 动作类型
        Returns:
            是否匹配
        """
        node_type = node_info.get("type")
        if not node_type:
            return False
        return action_type.startswith(node_type)
    
    def _check_entity_type_match(self, entity_info: Dict, action_type: str) -> bool:
        """
        检查实体类型是否与动作类型匹配
        Args:
            entity_info: 实体信息
            action_type: 动作类型
        Returns:
            是否匹配
        """
        entity_type = entity_info.get("type")
        if not entity_type:
            return False
        if entity_type == 'person' and action_type.startswith('author'):
            return True
        return False
    
    def _check_type(self, value: Any, expected_type_str: Optional[str]) -> bool:
        """
        检查值的类型是否符合预期
        Args:
            value: 需要检查的值
            expected_type_str: 预期类型的字符串表示
        Returns:
            是否符合预期类型
        """
        if not expected_type_str:
            return True
            
        expected_type_str = expected_type_str.lower()
        
        # 检查列表类型
        if expected_type_str.startswith("list[") and expected_type_str.endswith("]"):
            if not isinstance(value, list):
                return False
                
            inner_type_str = expected_type_str[5:-1] if len(expected_type_str) > 6 else "str"
            
            if inner_type_str == "str":
                return all(isinstance(item, str) for item in value)
            else:
                return True  # 默认假设其他列表类型匹配
                
        # 基本类型映射
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "number": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
            "object": object
        }
        
        target_py_type = type_map.get(expected_type_str)
        return isinstance(value, target_py_type) if target_py_type else True


class ToolQueryParamCompletion:
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
        self.adapter = GenericDatasetAdapter(debug=debug)
        
        # 创建框架
        self.framework = ParameterFillingFramework(
            tool_graph=tool_graph,
            tool_descriptions=tool_descriptions,
            adapter=self.adapter,
            param_dependency_path=dependency_graph_path,
            history_max_len=history_max_len,
            debug=debug
        )
        
        # 设置目标
        if goal:
            self.set_goal(goal)
    
    def set_goal(self, goal: str):
        """
        设置目标
        Args:
            goal: 目标文本
        """
        self.adapter.set_goal(goal)
        
    def record_execution(self, action_text: str, result: Any):
        """
        记录工具执行
        Args:
            action_text: 动作文本
            result: 执行结果
        """
        self.framework.record_execution(action_text, result)
    
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