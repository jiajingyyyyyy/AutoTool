from autool.core.tool_predict.datastruct import ToolGraph
from typing import Dict, List, Any, Tuple, Optional, Set
import os    
import abc
from .param_dependency import ParameterDependencyGraph, ExecutionHistory

class EnvironmentAdapter(abc.ABC):
    """
    Environment adapter base class, defines interfaces related to specific environments/datasets.
    Each new environment should implement a concrete adapter subclass.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the environment adapter
        Args:
            debug: Whether to enable debug mode
        """
        self.debug = debug
    
    @abc.abstractmethod
    def reset(self, init_observation: str = None):
        pass

    @abc.abstractmethod
    def parse_action(self, action_text: str, tool_descriptions: Dict) -> Optional[Dict[str, Any]]:
        """
        Parse action text and return structured action data
        Args:
            action_text: Action text
            tool_descriptions: Tool description dictionary
        Returns:
            Parsed action dictionary, format: {"tool_name": str, "inputs": Dict[str, Any]}
        """
        pass
    
    @abc.abstractmethod
    def infer_output(self, tool_name: str, inputs: Dict[str, Any], result: Any) -> Dict[str, Any]:
        """
        Infer structured output based on tool execution result
        Args:
            tool_name: Tool name
            inputs: Input parameters
            result: Execution result
        Returns:
            Inferred structured output dictionary
        """
        pass
    
    @abc.abstractmethod
    def update_state(self, action_parsed: Dict[str, Any], structured_outputs: Dict[str, Any]) -> None:
        """
        Update environment state based on action and output
        Args:
            action_parsed: Parsed action dictionary
            structured_outputs: Structured output dictionary
        """
        pass
    
    @abc.abstractmethod
    def get_contextual_params(self, action_type: str, missing_params: Set[str], required_params_info: Dict) -> Dict[str, Any]:
        """
        Infer parameter values based on current environment state
        Args:
            action_type: Target action type
            missing_params: Set of missing parameters
            required_params_info: Dictionary of required parameter descriptions
        Returns:
            Inferred parameter dictionary
        """
        pass

    @abc.abstractmethod
    def generate_action_from_params(self, action_type: str, params: Dict[str, Any]) -> str:
        pass
    
    def set_debug(self, debug: bool) -> None:
        """
        Set debug mode
        Args:
            debug: Whether to enable debug mode
        """
        self.debug = debug

    def get_state(self, state_name: str) -> Any:
        """
        Get specific state value
        Args:
            state_name: State name
        Returns:
            State value, or None if it doesn't exist
        """
        # Default implementation, subclasses can override this method to provide more complex state access
        return None

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


class CoreParameterFillingEngine:
    """
    Core parameter filling engine, implements general priority filling logic.
    This part of the code is environment-independent and responsible for integrating multiple information sources for parameter filling.
    """
    
    def __init__(self, debug: bool = False, tool_graph: ToolGraph = None):
        """
        初始化核心参数填充引擎
        Args:
            debug: 是否启用调试模式
        """
        self.tool_graph = tool_graph
        self.debug = debug
    
    def fill(self, 
             target_tool: str, 
             tool_graph: ToolGraph,
             param_graph: ParameterDependencyGraph,
             history: ExecutionHistory,
             adapter: EnvironmentAdapter,
             existing_params: Optional[Dict[str, Any]] = None,
             lookback_k: int = 5) -> Dict[str, Any]:
        """
        基于优先级策略填充参数
        Args:
            target_tool: 目标工具名称
            tool_graph: 工具调用图
            param_graph: 参数依赖图
            history: 执行历史
            adapter: 环境适配器实例
            existing_params: 已有的参数字典
            lookback_k: 历史回溯步数
        Returns:
            填充方式
            填充后的参数字典
        """
        meta_data = {}
        if self.debug: print(f"\n--- CoreEngine: Filling Params for '{target_tool}' ---")
        
        if target_tool not in tool_graph.nodes:
            if self.debug: print(f"Error: Target tool '{target_tool}' not found in tool graph.")
            return existing_params or {}
        
        # 获取工具所需参数信息
        tool_node = tool_graph.nodes[target_tool]
        required_params_info = tool_node.input_params  # 字典 {'name': {'type':...}}
        if self.debug: print(f'Required Params Info: {required_params_info}')
        # 初始化填充参数和缺失参数
        filled_params = existing_params.copy() if existing_params else {}
        missing_params = set(required_params_info.keys()) - set(filled_params.keys())
        
        if not missing_params:
            if self.debug: print("All parameters already provided.")
            return meta_data, filled_params
            
        if self.debug: print(f"Required: {set(required_params_info.keys())}, Missing: {missing_params}")
        
        # --- 填充优先级 ---
        
        # 1. 从参数依赖图填充
        if missing_params:
            dep_graph_filled = self._fill_from_dependency_graph(
                target_tool, 
                missing_params.copy(), 
                required_params_info, 
                param_graph, 
                history, 
                lookback_k
            )
            meta_data["PDG_filled"] = dep_graph_filled
            filled_params.update(dep_graph_filled)
            missing_params -= set(dep_graph_filled.keys())
            if dep_graph_filled: 
                print(f"Filled from Dependency Graph: {dep_graph_filled}")
        
        # 2. 从环境上下文状态填充
        if missing_params:
            context_filled = adapter.get_contextual_params(
                target_tool, 
                missing_params.copy(), 
                required_params_info
            )
            filled_params.update(context_filled)
            missing_params -= set(context_filled.keys())
            if context_filled: 
                print(f"Filled from Context: {context_filled}")
            meta_data["context_filled"] = context_filled
        
        # 最终检查还有哪些参数未填充
        final_missing = set(required_params_info.keys()) - set(filled_params.keys())
        if final_missing and self.debug:
            print(f"Warning: Could not fill all params for '{target_tool}'. Missing: {final_missing}")
            
        return meta_data, filled_params
   

    def _fill_from_dependency_graph(self, 
                               target_tool: str, 
                               missing_params: Set[str], 
                               target_params_info: Dict,
                               param_graph: ParameterDependencyGraph,
                               history: ExecutionHistory,
                               lookback_k: int,
                               tool_patern_ROI=3) -> Dict[str, Any]:
        """
        从参数依赖图填充参数（智能处理列表和单值类型）
        """
        filled_params = {}

        # 记录已用过的单值，避免重复
        used_values = set(filled_params.values())

        # --- new feat ---
        # 先从匹配到的当前工具链中的历史参数填充情况进行填充
        matched_tool_sequences = history.get_latest_records(tool_patern_ROI)
        print(f'[matched_tool_sequences]: {matched_tool_sequences}')
        # 要获取所查找工具链的工具名称
        matched_tool_sequences = [seq.get("tool_name") for seq in matched_tool_sequences]
        print(f'[matched_tool_sequences]: {matched_tool_sequences}')

        for param_name in list(missing_params):
            potential_sources = self.tool_graph.get_param_sources(matched_tool_sequences, target_tool, param_name)
            # 这里是直接赋值，后面要改
            print(f'[potential_sources]: {potential_sources}')
            param_type = target_params_info.get(param_name, {}).get('type')
            
            for source_tool, source_param, count in potential_sources:
                
                source_value = history.get_output_value(source_tool, source_param, lookback_k)
                print(f'[source_value]: {source_value}')
                if source_value is not None:
                    # 源是列表，目标是单值
                    if isinstance(source_value, list) and (param_type in ['str', 'int', 'float', None]):
                        # 选择未被素用过的元
                        candidate = next((v for v in source_value if v not in used_values), None)
                        if candidate is not None and self._check_type(candidate, param_type):
                            filled_params[param_name] = candidate
                            used_values.add(candidate)
                            missing_params.remove(param_name)
                            if self.debug:
                                print(f"PDG Fill: '{param_name}' <- '{source_tool}.{source_param}[{source_value.index(candidate)}]' (Count:{count})")
                            break
                    # 类型一致或目标是列表，直接赋值
                    elif self._check_type(source_value, param_type):
                        filled_params[param_name] = source_value
                        if isinstance(source_value, (str, int, float)):
                            used_values.add(source_value)
                        missing_params.remove(param_name)
                        if self.debug:
                            print(f"PDG Fill: '{param_name}' <- '{source_tool}.{source_param}' (Count:{count})")
                        break

        # 再按照当前参数整体填充情况，按照填充频率从高到低进行填充
        for param_name in list(missing_params):
            potential_sources = param_graph.get_potential_sources(target_tool, param_name)
            # 这里是直接赋值，后面要改
            param_type = target_params_info.get(param_name, {}).get('type')
            
            for source_tool, source_param, count in potential_sources:
                
                source_value = history.get_output_value(source_tool, source_param, lookback_k)
                if source_value is not None:
                    # 源是列表，目标是单值
                    if isinstance(source_value, list) and (param_type in ['str', 'int', 'float', None]):
                        # 选择未被素用过的元
                        candidate = next((v for v in source_value if v not in used_values), None)
                        if candidate is not None and self._check_type(candidate, param_type):
                            filled_params[param_name] = candidate
                            used_values.add(candidate)
                            missing_params.remove(param_name)
                            if self.debug:
                                print(f"PDG Fill: '{param_name}' <- '{source_tool}.{source_param}[{source_value.index(candidate)}]' (Count:{count})")
                            break
                    # 类型一致或目标是列表，直接赋值
                    elif self._check_type(source_value, param_type):
                        filled_params[param_name] = source_value
                        if isinstance(source_value, (str, int, float)):
                            used_values.add(source_value)
                        missing_params.remove(param_name)
                        if self.debug:
                            print(f"PDG Fill: '{param_name}' <- '{source_tool}.{source_param}' (Count:{count})")
                        break

        return filled_params


    def _fill_from_history_fallback(self, 
                                   target_tool: str, 
                                   missing_params: Set[str], 
                                   target_params_info: Dict,
                                   history: ExecutionHistory,
                                   lookback_k: int) -> Dict[str, Any]:
        """
        从历史记录回退填充参数（基于参数名和类型匹配）
        Args:
            target_tool: 目标工具名称
            missing_params: 缺失的参数集合
            target_params_info: 必需的参数描述字典
            history: 执行历史
            lookback_k: 历史回溯步数
        Returns:
            填充的参数字典
        """
        filled_params = {}
        latest_records = history.get_latest_records(lookback_k)
        
        if not latest_records:
            return filled_params
            
        for param_name in list(missing_params):
            param_info = target_params_info.get(param_name, {})
            param_type = param_info.get('type')
            found = False
            
            for record in reversed(latest_records):
                tool_name = record.get('tool_name')
                outputs = record.get('outputs', {})
                
                if not tool_name or not outputs:
                    continue
                    
                # 检查输出参数
                for output_name, output_value in outputs.items():
                    if output_value is None:
                        continue
                        
                    if self._check_type(output_value, param_type) and self._is_param_match(output_name, param_name, param_type):
                        filled_params[param_name] = output_value
                        missing_params.remove(param_name)
                        if self.debug:
                            print(f"History Fill: '{param_name}' <- '{tool_name}.{output_name}'")
                        found = True
                        break
                
                if found:
                    break
                    
                # 检查输入参数
                inputs = record.get('inputs', {})
                if not found and inputs:
                    for input_name, input_value in inputs.items():
                        if input_value is None:
                            continue
                            
                        if self._check_type(input_value, param_type) and self._is_param_match(input_name, param_name, param_type):
                            filled_params[param_name] = input_value
                            missing_params.remove(param_name)
                            if self.debug:
                                print(f"History Fill: '{param_name}' <- '{tool_name}.{input_name}'")
                            found = True
                            break
                            
                if found:
                    break
        
        return filled_params
    
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
    
    def _is_param_match(self, name1: str, name2: str, param_type: Optional[str] = None) -> bool:
        """
        判断两个参数名是否匹配
        Args:
            name1: 第一个参数名
            name2: 第二个参数名
            param_type: 参数类型
        Returns:
            是否匹配
        """
        # 精确名称匹配
        if name1 == name2:
            return True
            
        # 规范化参数名
        n1_norm = name1.lower().replace("_", "").replace("-", "")
        n2_norm = name2.lower().replace("_", "").replace("-", "")
        
        if n1_norm == n2_norm:
            return True
            
        # 常见参数别名集合
        node_aliases = {"node", "id", "node_id", "node1", "node2", "paper_id", "author_id", "movie_id", "person_id"}
        graph_aliases = {"graph", "graph_type", "network_type"}
        location_aliases = {"location", "city", "latitude", "longitude", "current_location"}
        name_aliases = {"name", "movie_name", "person_name", "author_name", "object_name"}
        object_aliases = {"obj", "object", "item", "thing", "target", "taken_object"}
        state_aliases = {"state", "status", "receptacle_state"}
        content_aliases = {"content", "observed_content", "result", "result_value"}
        
        # 检查是否在同一个别名集合中
        alias_sets = [node_aliases, graph_aliases, location_aliases, name_aliases, object_aliases, state_aliases, content_aliases]
        
        for alias_set in alias_sets:
            if n1_norm in alias_set and n2_norm in alias_set:
                return True
                
        # 包含关系检查
        if n1_norm in n2_norm or n2_norm in n1_norm:
            return True
            
        return False
    
    def set_debug(self, debug: bool) -> None:
        """
        设置调试模式
        Args:
            debug: 是否启用调试模式
        """
        self.debug = debug


class ParameterFillingFramework:
    """
    Parameter filling framework, providing a unified parameter filling interface
    and coordinating internal components.
    """
    
    def __init__(self, 
                tool_graph: ToolGraph,
                tool_descriptions: Dict[str, Dict],
                adapter: EnvironmentAdapter,
                param_dependency_path: Optional[str] = None,
                history_max_len: int = 20,
                debug: bool = False):
        """
        Initialize the parameter filling framework
        Args:
            tool_graph: Tool calling graph
            tool_descriptions: Dictionary of tool descriptions
            adapter: Environment adapter instance
            param_dependency_path: Path to parameter dependency graph file
            history_max_len: Maximum length of execution history
            debug: Whether to enable debug mode
        """
        self.debug = debug
        self.tool_graph = tool_graph
        self.adapter = adapter
        
        # 设置所有组件的调试模式
        self.adapter.set_debug(debug)
        
        # 初始化参数依赖图
        self.param_graph = ParameterDependencyGraph()
        if param_dependency_path and os.path.exists(param_dependency_path):
            try:
                self.param_graph.load_from_file(param_dependency_path)
                if self.debug:
                    print(f"Successfully loaded parameter dependency graph from {param_dependency_path}")
                    stats = self.param_graph.get_stats()
                    print(f"Dependency graph stats: {stats}")
            except Exception as e:
                if self.debug:
                    print(f"Failed to load parameter dependency graph from {param_dependency_path}: {e}")
        elif self.debug:
            print("No valid parameter dependency graph path provided, using empty graph.")
        self.param_graph.set_debug(debug)
        
        # 初始化执行历史
        self.history = ExecutionHistory(max_len=history_max_len)
        
        # 初始化核心引擎
        self.core_engine = CoreParameterFillingEngine(debug=debug, tool_graph=self.tool_graph)
    

    def record_execution(self, action_text, action_parsed: str, result: Any) -> None:
        """
        Record tool execution
        Args:
            action_text: Action text
            result: Execution result
        """
        if not action_parsed:
            if self.debug:
                print("Failed to parse action.")
            return
            
        tool_name = action_parsed.get("tool_name")
        inputs = action_parsed.get("inputs", {})
        
        if not tool_name:
            if self.debug:
                print("No tool name found in parsed action.")
            return
            
        # 推断结构化输出
        structured_outputs = self.adapter.infer_output(tool_name, inputs, result)
        
        # 更新环境状态
        self.adapter.update_state(action_parsed, structured_outputs)
        
        # 添加到执行历史
        record = {
            "tool_name": tool_name,
            "inputs": inputs,
            "outputs": structured_outputs,
            "raw_action": action_text,
            "raw_result": result
        }
        self.history.add_record(record)
        
        if self.debug:
            print(f"--- End Recording Execution ---")
    
    def fill_parameters(self, target_tool: str, existing_params: Optional[Dict[str, Any]] = None, lookback_k: int = 5) -> Dict[str, Any]:
        """
        Fill parameters
        Args:
            target_tool: Target tool name
            existing_params: Dictionary of existing parameters
            lookback_k: Number of history steps to look back
        Returns:
            Dictionary of filled parameters
        """
        return self.core_engine.fill(
            target_tool=target_tool,
            tool_graph=self.tool_graph,
            param_graph=self.param_graph,
            history=self.history,
            adapter=self.adapter,
            existing_params=existing_params,
            lookback_k=lookback_k
        )
    
    def set_debug(self, debug: bool) -> None:
        """
        设置调试模式
        Args:
            debug: 是否启用调试模式
        """
        self.debug = debug
        self.adapter.set_debug(debug)
        self.param_graph.set_debug(debug)
        self.core_engine.set_debug(debug)

class ParamCompletion:
    """ 参数自动填充 """
    def __init__(self, tool_graph: ToolGraph):
        """
        初始化参数自动填充器
        Args:
            tool_graph: 工具调用图
        """
        self.tool_graph = tool_graph
        self.executed_tools = []  # 记录已执行的工具序列
        self.tool_outputs = {}    # 存储工具执行结果 {工具名称: {参数名: 参数值}}
        self.debug = False        # 调试模式

    def complete_params_with_inertial_chain(self, current_tool: str, next_tool: str, params: dict = None) -> Dict[str, Any]:
        """
        基于惯性调用链的参数自动填充
        Args:
            current_tool: 当前工具名称
            next_tool: 下一个（惯性调用）工具名称
            params: 已有的参数字典
        Returns:
            填充后的参数字典
        """
        if self.debug:
            print(f"开始基于惯性调用链填充 {next_tool} 的参数")
            
        if next_tool not in self.tool_graph.nodes:
            if self.debug:
                print(f"工具 {next_tool} 不在工具图中")
            return {}
            
        # 获取需要填充的目标参数
        next_tool_node = self.tool_graph.nodes[next_tool]
        required_params = next_tool_node.input_params
        
        # 初始化已填充参数
        filled_params = params.copy() if params else {}
        
        # 找出缺失的参数
        missing_params = set(required_params.keys()) - set(filled_params.keys())
        if not missing_params:
            if self.debug:
                print(f"工具 {next_tool} 的所有参数已填充")
            return filled_params
            
        if self.debug:
            print(f"工具 {next_tool} 缺少参数: {missing_params}")
        
        # 首先检查历史执行链中的工具
        historically_filled = self._fill_from_execution_history(next_tool, missing_params, required_params)
        filled_params.update(historically_filled)
        
        # 更新剩余缺失的参数
        missing_params = set(required_params.keys()) - set(filled_params.keys())
        if not missing_params:
            if self.debug:
                print(f"已从历史执行链填充所有参数")
            return filled_params
            
        # 分析惯性调用链上的参数依赖
        inertial_filled = self._fill_from_inertial_chain(current_tool, next_tool, missing_params, required_params)
        filled_params.update(inertial_filled)
        
        # 最终检查还有哪些参数未填充
        missing_params = set(required_params.keys()) - set(filled_params.keys())
        if missing_params and self.debug:
            print(f"无法填充的参数: {missing_params}")
            
        return filled_params
    
    def _fill_from_execution_history(self, tool_name: str, missing_params: set, required_params: dict) -> dict:
        """
        从历史执行的工具中填充参数
        Args:
            tool_name: 当前工具名称
            missing_params: 缺失的参数集合
            required_params: 必需的参数描述字典
        Returns:
            填充的参数字典
        """
        filled_params = {}
        
        # 从最近执行的工具往前检查
        for executed_tool in reversed(self.executed_tools):
            if executed_tool not in self.tool_outputs: # 有输出结果
                continue
                
            # 检查此工具的输出参数
            tool_outputs = self.tool_outputs[executed_tool]
            
            for param_name in list(missing_params):
                param_info = required_params[param_name]
                param_type = param_info.get("type", "")
                param_desc = param_info.get("description", "")
                
                # 检查输出结果中是否有匹配项
                if "result" in tool_outputs:
                    result = tool_outputs["result"]
                    # 直接匹配参数名
                    if isinstance(result, dict) and param_name in result:
                        filled_params[param_name] = result[param_name]
                        missing_params.remove(param_name) # 完成一次参数填充
                        if self.debug:
                            print(f"从 {executed_tool} 的输出结果中找到参数 {param_name}")
                        continue
                
                # 检查输入参数中是否有匹配项
                if "inputs" in tool_outputs:
                    inputs = tool_outputs["inputs"]
                    if isinstance(inputs, dict):
                        for input_name, input_value in inputs.items():
                            if self._is_param_match(input_name, param_name, param_type):
                                filled_params[param_name] = input_value
                                missing_params.remove(param_name)
                                if self.debug:
                                    print(f"从 {executed_tool} 的输入参数 {input_name} 中找到参数 {param_name}")
                                break
            
            # 如果所有参数都已填充，提前结束
            if not missing_params:
                break
        
        return filled_params
    
    def _fill_from_inertial_chain(self, current_tool: str, next_tool: str, missing_params: set, required_params: dict) -> dict:
        """
        从惯性调用链中填充参数
        Args:
            current_tool: 当前工具名称
            next_tool: 下一个工具名称
            missing_params: 缺失的参数集合
            required_params: 必需的参数描述字典
        Returns:
            填充的参数字典
        """
        filled_params = {}
        
        # 查找共同的历史调用链
        common_paths = self._find_common_paths(current_tool, next_tool)
        
        if not common_paths and self.debug:
            print(f"未找到 {current_tool} 和 {next_tool} 的共同调用链")
            return filled_params
            
        # 检查每条路径
        for path in common_paths:
            # 找到当前工具在路径中的位置
            if current_tool not in path:
                continue
            
            # 当路径包含多个相同元素时，返回第一次出现的索引
            current_tool_index = path.index(current_tool)
            
            # 提取当前工具之前的所有工具
            previous_tools = path[:current_tool_index+1]
            
            # 从最近的工具开始检查
            for prev_tool in reversed(previous_tools):
                if prev_tool not in self.tool_graph.nodes:
                    continue
                    
                prev_tool_node = self.tool_graph.nodes[prev_tool]
                
                # 检查此工具的参数
                for param_name in list(missing_params):
                    param_info = required_params[param_name]
                    param_type = param_info.get("type", "")
                    
                    # 检查前继工具的输出参数是否匹配
                    for prev_param_name, prev_param_node in prev_tool_node.param_graph.nodes.items():
                        if prev_param_node.is_output and self._is_param_match(prev_param_name, param_name, param_type):
                            # 如果历史执行中有此工具的执行结果，使用它
                            if prev_tool in self.tool_outputs and "result" in self.tool_outputs[prev_tool]:
                                result = self.tool_outputs[prev_tool]["result"]
                                if isinstance(result, dict) and prev_param_name in result:
                                    filled_params[param_name] = result[prev_param_name]
                                    missing_params.remove(param_name)
                                    if self.debug:
                                        print(f"从惯性调用链中的 {prev_tool} 找到匹配参数 {param_name}")
                                    break
            
            # 如果所有参数都已填充，提前结束
            if not missing_params:
                break
        
        return filled_params
    
    def _find_common_paths(self, current_tool: str, next_tool: str) -> List[List[str]]:
        """
        查找包含当前工具和下一个工具的共同路径
        Args:
            current_tool: 当前工具名称
            next_tool: 下一个工具名称
        Returns:
            共同路径列表
        """
        common_paths = []
        
        # 获取包含两个工具的所有路径
        if current_tool in self.tool_graph.path_index and next_tool in self.tool_graph.path_index:
            current_tool_paths = self.tool_graph.path_index[current_tool]
            next_tool_paths = self.tool_graph.path_index[next_tool]
            
            # 找出共同路径
            common_path_indices = current_tool_paths.intersection(next_tool_paths)
            
            for path_idx in common_path_indices:
                if path_idx < len(self.tool_graph.paths):
                    common_paths.append(self.tool_graph.paths[path_idx].tools)
        
        return common_paths
    
    def _is_param_match(self, source_param: str, target_param: str, target_type: str) -> bool:
        """
        判断参数是否匹配
        Args:
            source_param: 源参数名
            target_param: 目标参数名
            target_type: 目标参数类型
        Returns:
            是否匹配
        """
        # 精确名称匹配
        if source_param == target_param:
            return True
            
        # 规范化名称后匹配
        norm_source = self._normalize_param_name(source_param)
        norm_target = self._normalize_param_name(target_param)
        
        if norm_source == norm_target:
            return True
            
        # 检查常见参数命名模式
        if self._check_common_param_patterns(source_param, target_param):
            return True
            
        return False
    
    def _normalize_param_name(self, param_name: str) -> str:
        """
        规范化参数名，移除常见前缀后缀，统一格式
        Args:
            param_name: 参数名
        Returns:
            规范化后的参数名
        """
        # 转为小写
        norm_name = param_name.lower()
        
        # 移除下划线、连字符等
        norm_name = norm_name.replace("_", "").replace("-", "")
        
        # 移除常见前缀
        prefixes = ["param", "arg", "input", "in", "the", "a", "an"]
        for prefix in prefixes:
            if norm_name.startswith(prefix):
                norm_name = norm_name[len(prefix):]
                
        # 移除常见后缀
        suffixes = ["param", "arg", "value", "type", "id", "name"]
        for suffix in suffixes:
            if norm_name.endswith(suffix):
                norm_name = norm_name[:-len(suffix)]
                
        return norm_name
    
    def _check_common_param_patterns(self, source_param: str, target_param: str) -> bool:
        """
        检查常见的参数命名模式
        Args:
            source_param: 源参数名
            target_param: 目标参数名
        Returns:
            是否符合常见命名模式
        """
        # 例如：file_path 和 filepath, file_name 和 filename
        if (source_param.replace("_", "") == target_param or 
            target_param.replace("_", "") == source_param):
            return True
            
        # 例如：itemID 和 item_id
        source_lower = source_param.lower()
        target_lower = target_param.lower()
        
        if source_lower == target_lower:
            return True
        
        # 组合检查，如 fileId 和 file_id
        source_no_under = source_lower.replace("_", "")
        target_no_under = target_lower.replace("_", "")
        
        return source_no_under == target_no_under

    def record_tool_execution(self, tool_name: str, inputs: Dict[str, Any], result: Any) -> None:
        """
        记录工具执行结果
        Args:
            tool_name: 工具名称
            inputs: 工具输入参数
            result: 工具执行结果
        """
        self.executed_tools.append(tool_name)
        self.tool_outputs[tool_name] = {
            "inputs": inputs,
            "result": result
        }
        
        # 更新参数图的值示例
        if tool_name in self.tool_graph.nodes:
            tool_node = self.tool_graph.nodes[tool_name]
            
            # 为输入参数添加值示例
            for param_name, value in inputs.items():
                if param_name in tool_node.param_graph.nodes:
                    tool_node.param_graph.nodes[param_name].add_value_example(value)
            
            # 为输出参数添加值示例
            if "result" in tool_node.param_graph.nodes and result is not None:
                tool_node.param_graph.nodes["result"].add_value_example(result)

    def _is_similar_name(self, name1: str, name2: str) -> bool:
        """
        判断两个参数名是否相似
        Args:
            name1: 第一个参数名
            name2: 第二个参数名
        Returns:
            是否相似
        """
        # 将参数名规范化（转换为小写，移除下划线等）
        n1 = name1.lower().replace("_", "").replace("-", "")
        n2 = name2.lower().replace("_", "").replace("-", "")
        
        # 检查是否包含关系
        return n1 in n2 or n2 in n1
    
    def set_debug(self, debug: bool) -> None:
        """
        设置调试模式
        Args:
            debug: 是否启用调试模式
        """
        self.debug = debug
