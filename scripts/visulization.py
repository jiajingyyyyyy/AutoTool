from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Set
import os
import json
from autool.utils.embedding import get_embedding, compute_similarity
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.patheffects as path_effects # <--- 添加这一行

class ExecutionHistory:
    """存储结构化的工具执行历史记录"""
    def __init__(self, max_len: int = 20):
        """
        初始化执行历史
        Args:
            max_len: 最大历史记录条数
        """
        self.history: List[Dict[str, Any]] = []
        self.max_len = max_len
        self.debug = False # 添加 debug 标志

    def set_debug(self, debug: bool):
        self.debug = debug

    def add_record(self, record: Dict[str, Any]):
        """
        添加一条结构化记录到历史中。
        记录应包含 'tool_name', 'inputs', 'outputs'。
        """
        # 可以添加一些验证确保记录格式正确
        if "tool_name" in record and "inputs" in record and "outputs" in record:
            self.history.append(record)
            # 如果历史超过最大长度，移除最旧的记录
            if len(self.history) > self.max_len:
                self.history.pop(0)
            if self.debug: print(f"  History: Added record for '{record['tool_name']}'. Size: {len(self.history)}")
        elif self.debug:
             print(f"Warning: Attempted to add invalid record to history: {record}")


    def get_latest_records(self, k: int = 5) -> List[Dict[str, Any]]:
        """
        获取最近的 k 条历史记录。
        """
        return self.history[-k:]

    def get_output_value(self, tool_name: str, output_param_name: str, lookback: int = 5) -> Optional[Any]:
        """
        从最近的执行记录中查找特定工具的特定输出参数的值。
        Args:
            tool_name: 要查找的工具名称。
            output_param_name: 要查找的输出参数名称。
            lookback: 向前回溯查找的最大步数。
        Returns:
            找到的参数值，如果未找到则返回 None。
        """
        # 从最近的记录开始反向查找
        relevant_history = self.history[-lookback:]
        for record in reversed(relevant_history):
            if record.get("tool_name") == tool_name:
                outputs = record.get("outputs", {})
                if output_param_name in outputs:
                    value = outputs[output_param_name]
                    # 不返回 None 值，除非它是唯一的匹配项？ (目前返回 None)
                    # if value is not None:
                    #     return value
                    return value # 直接返回值，即使是 None
        return None # 遍历完未找到 
import time

class ParamEdge:
    """参数依赖边，表示一个工具的输出参数与另一个工具的输入参数之间的依赖关系"""
    def __init__(self, source_tool: str, source_param: str, target_tool: str, target_param: str) -> None:
        self.source_tool = source_tool      # 源工具名称
        self.source_param = source_param    # 源工具输出参数名称
        self.target_tool = target_tool      # 目标工具名称
        self.target_param = target_param    # 目标工具输入参数名称
        self.count = 1                      # 该依赖关系出现的次数

    def increment_count(self) -> None:
        """增加依赖计数"""
        self.count += 1

class ParamNode:
    """参数节点，表示工具的一个参数"""
    def __init__(self, param_name: str, param_type: str = "", param_desc: str = "") -> None:
        self.name = param_name          # 参数名称
        self.type = param_type          # 参数类型
        self.description = param_desc   # 参数描述
        self.is_output = False          # 是否为输出参数
        self.value_examples = []        # 参数值示例
        self.value_cache_size = 2  # 参数值的缓存大小

    def set_as_output(self) -> None:
        """将参数标记为输出参数"""
        self.is_output = True
    
    def add_value_example(self, value: Any) -> None:
        """添加参数值示例"""
        if len(self.value_examples) < self.value_cache_size:  # 最多保存5个示例
            self.value_examples.append(value)

        
class ParamGraph:
    """参数依赖图，表示工具内部参数之间的依赖关系"""
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name            # 所属工具名称
        self.nodes: Dict[str, ParamNode] = {} # 参数节点集合
        self.returns: Dict[str, ParamNode] = {} # 返回值节点集合
    def add_param_node(self, param_name: str, param_type: str = "", param_desc: str = "") -> None:
        """添加参数节点"""
        self.nodes[param_name] = ParamNode(param_name, param_type, param_desc)
    def add_param_return(self, return_name: str, return_type: str = "", return_desc: str = "") -> None:
        """添加返回值节点"""
        self.returns[return_name] = ParamNode(return_name, return_type, return_desc).set_as_output()

class ToolNode:
    """工具节点，表示图中的一个工具"""
    def __init__(self, tool_name: str, tool_desc: dict) -> None:
        self.name = tool_name
        self.description = tool_desc.get("tool_desc", "")
        self.args = tool_desc.get("args", [])
        self.returns = tool_desc.get("returns", {})
        # 存储工具的输入参数信息
        self.input_params = {}
        self.output_params = {}
        for arg in self.args:
            arg_name = arg.get("arg_name", "")
            arg_type = arg.get("arg_type", "")
            arg_desc = arg.get("arg_desc", "")
            self.input_params[arg_name] = {
                "type": arg_type,
                "description": arg_desc
            }
        for output in self.returns:
            output_name = output.get("arg_name", "")
            output_type = output.get("arg_type", "")
            output_desc = output.get("arg_desc", "")
            self.output_params[output_name] = {
                "type": output_type,
                "description": output_desc
            }
        
        # 添加参数子图
        self.param_graph = ParamGraph(tool_name)
        
        # 初始化参数子图的节点
        for arg_name, arg_info in self.input_params.items():
            self.param_graph.add_param_node(
                arg_name, 
                arg_info.get("type", ""), 
                arg_info.get("description", "")
            )
        
        # 添加默认输出参数节点
        self.param_graph.add_param_return("raw_result", "object", "工具的输出结果")
        for output_name, output_info in self.output_params.items():
            self.param_graph.add_param_return(
                output_name, 
                output_info.get("type", ""), 
                output_info.get("description", "")
            )
        
    
    def format_tool_description(self) -> str:
        """格式化工具描述，与toolkit中的格式保持一致"""
        # 构建工具名称和描述
        tool_desc = f"{self.name}: {self.description}\n"
        
        # 添加参数描述
        for arg in self.args:
            arg_name = arg.get("arg_name", "")
            arg_type = arg.get("arg_type", "")
            arg_desc = arg.get("arg_desc", "")
            tool_desc += f"{arg_name}({arg_type}): {arg_desc}\n"
        
        # 添加额外空行
        tool_desc += "\n"
        
        return tool_desc

class ToolEdge:
    """工具调用边，连接两个工具节点，表示工具调用链中的一条边"""
    def __init__(self, source: str, target: str) -> None:
        self.source = source  # 源工具名称
        self.target = target  # 目标工具名称
        self.call_count = 0   # 调用次数
        self.param_mappings = {}  # 参数映射关系 {target_param: source_output_field}
    
    def increment_call(self) -> None:
        """增加调用计数"""
        self.call_count += 1
    
    def add_param_mapping(self, target_param: str, source_field: str) -> None:
        """添加参数映射关系"""
        self.param_mappings[target_param] = source_field

class ToolPath:
    """工具调用路径，表示一条完整的工具调用链"""
    def __init__(self, tools: List[str]) -> None:
        self.tools = tools  # 工具序列
        self.frequency = 1   # 路径出现频率
        self.confidence = 0.0 # 路径置信度
    
    def __eq__(self, other):
        """重写等号操作符，使得可以直接比较两个ToolPath对象是否具有相同的工具序列"""
        if isinstance(other, ToolPath):
            return self.tools == other.tools
        elif isinstance(other, list):
            return self.tools == other
        return False
    
    def __hash__(self):
        """实现哈希方法，使ToolPath可以作为字典的键或集合的元素"""
        return hash(tuple(self.tools))
    
    def update_frequency(self, weight) -> None:
        """增加路径频率"""
        self.frequency += weight
    
    def match_prefix(self, prefix: List[str]) -> bool:
        """检查是否匹配前缀序列"""
        if len(prefix) > len(self.tools):
            return False
        return self.tools[:len(prefix)] == prefix
    
    def get_subsequent_path(self, prefix_length: int) -> List[str]:
        """获取前缀之后的路径"""
        if prefix_length >= len(self.tools):
            return []
        return self.tools[prefix_length:]


class ToolGraph:
    """工具调用图，使用工具作为节点，工具调用链作为边"""
    def __init__(self) -> None:
        # 基础结构
        self.nodes: Dict[str, ToolNode] = {}  # 工具节点字典 {tool_name: ToolNode}
        self.edges: Dict[str, Dict[str, ToolEdge]] = {}  # 工具调用边字典 {source: {target: ToolEdge}}
        
        # 路径管理
        self.paths: List[ToolPath] = []  # 工具调用路径集合，每个唯一路径只存储一次
        self.tooll2chainID: Dict[str, set[int]] = defaultdict(set)  # 索引：工具名到路径ID的映射 {tool_name: {path_id1, path_id2, ...}}
        self.chain2ID: Dict[Tuple[str, ...], int] = {}  # 查找表：路径序列到路径ID的映射，用于快速查找路径是否已存在
        
        # 新增：参数依赖边存储
        # 结构: {target_tool: {target_param: {(source_tool, source_param): ParamEdge}}}
        self.param_edges: Dict[str, Dict[str, Dict[Tuple[str, str], ParamEdge]]] = defaultdict(lambda: defaultdict(dict))
        
        # 元数据
        self.task_description = ""
        self.previous_progress = ""
        self.debug = False
        # self.load_from_json("tool_description.json", "cleaned_sequences.json")
        # if self.debug: print(f'[DEBUG INFO]: load from json: {tool_description}, {tool_trajectory}')
        # self.load_from_json(tool_description, tool_trajectory)
        # print(f'已加载路径索引: {len(self.tooll2chainID)}个工具，{len(self.paths)}条路径')
        # print(f"Tool Graph Init Done !")
        # for k, v in self.tooll2chainID.items():
        #     print(f'debug on path_index {k}: {v}')
        # 初始化父路径索引
    def add_node(self, tool_name: str, tool_desc: dict) -> None:
        """添加工具结点，同时初始化该工具的出边字典"""
        self.nodes[tool_name] = ToolNode(tool_name, tool_desc)
        # 初始化该工具的出边字典
        if tool_name not in self.edges:
            self.edges[tool_name] = {}
            self.tooll2chainID[tool_name] = set() 
    
    def add_edge(self, source: str, target: str) -> None:
        """添加工具调用边，同时更新路径索引"""
        if source not in self.nodes:
            raise ValueError(f"in add_edge, source tool {source} not found")
        if target not in self.nodes:
            raise ValueError(f"in add_edge, target tool {target} not found")
        
        if target not in self.edges[source]:
            self.edges[source][target] = ToolEdge(source, target)
    
    # --- might be dispaired, use update graph instead
    def record_tool_sequence(self, tool_sequence: List[str], weight=1) -> None:
        """这里传入的是一个完整的工具调用序列，更新工具调用链和工具之间调用边的信息， 用于出现了一次成功调用，需要更新图时"""
        if not tool_sequence:
            return
        # print('[DEBUG INFO]: call record_tool_sequence to update tool graph') 
        if len(tool_sequence) < 3:
            # print('[DEBUG INFO]: tool_sequence length < 3, return')
            return
        # 更新边的调用计数
        for i in range(len(tool_sequence) - 1):
            source = tool_sequence[i]
            target = tool_sequence[i + 1]
            
            # 如果边不存在，则创建
            if source not in self.edges or target not in self.edges.get(source, {}):
                # 如果点不存在，则创建
                if source not in self.nodes:
                    self.add_node(source, {}) # 允许没有见过的工具
                    print(f'[DEBUG INFO]: add new node: {source}')
                if target not in self.nodes:
                    self.add_node(target, {}) # 允许没有见过的工具
                    print(f'[DEBUG INFO]: add new node: {target}')
                self.add_edge(source, target)
            
            # 更新边的调用计数
            self.edges[source][target].increment_call()
        
    # 更新路径信息
        self._update_path(tool_sequence, weight=weight)

    def _update_path(self, tool_sequence: List[str], weight=1) -> None:
        """更新路径信息"""
        # 检查路径是否已存在
        tool_sequence_tuple = tuple(tool_sequence)
        
        print(f'[DEBUG INFO]: update with tool sequence: {tool_sequence}, weight: {weight}')
        if tool_sequence_tuple in self.chain2ID:
            # 路径已存在，更新频率
            path_id = self.chain2ID[tool_sequence_tuple]
            self.paths[path_id].update_frequency(weight)
            return
        
        # 添加新路径
        path_id = len(self.paths)
        new_path = ToolPath(tool_sequence)
        self.paths.append(new_path)
        self.chain2ID[tool_sequence_tuple] = path_id
        
        # 更新索引，将此路径ID添加到路径中每个工具的索引集合中
        for tool in tool_sequence:
            self.tooll2chainID[tool].add(path_id)
        
        # print(f"添加新路径: {tool_sequence}, 索引: {path_id}")
    
    def get_next_tools(self, current_tool: str) -> List[Tuple[str, int]]:
        """根据工具之间的直接连接，获取当前工具可能的下一个工具及其调用次数"""
        if current_tool not in self.edges:
            return []
        
        next_tools = []

        for target, edge in self.edges[current_tool].items():
            next_tools.append((target, edge.call_count))
        
        # 按调用次数降序排序
        next_tools.sort(key=lambda x: x[1], reverse=True)
        return next_tools

    def get_previous_tools(self, current_tool: str) -> List[Tuple[str, int]]:
        """根据工具之间的直接连接，获取当前工具可能的前置工具及其调用次数"""
        previous_tools = []
        
        # 遍历所有边，查找指向当前工具的边
        for source, targets in self.edges.items():
            if current_tool in targets:
                previous_tools.append((source, targets[current_tool].call_count))
        
        # 按调用次数降序排序
        previous_tools.sort(key=lambda x: x[1], reverse=True)
        return previous_tools
    
    def match_current_sequence(self, current_sequence: List[str]) -> List[Tuple[ToolPath, int]]:
        """
        基于工具链匹配的预测
        Args:
            current_sequence: 当前工具调用序列
        Returns:
            匹配的路径及其匹配长度列表 [(path, match_length)]
        """
        if not current_sequence:
            return []
        # 获取当前序列中最后一个工具相关的所有路径
        last_tool = current_sequence[-1]
        path_indices = self.tooll2chainID.get(last_tool, set())
        
        matched_paths = []
        total_frequency = 1
        for idx in path_indices: # 遍历所有可能的工具调用链
            path = self.paths[idx]
            # 找到当前序列在路径中的位置
            match_positions = self._find_subsequence_positions(current_sequence, path.tools)
            # print(f'match_position : {match_positions}')
            if match_positions: # 位置列表不为空
                # 使用最后一个匹配位置（最新的匹配），因为在实际情况中，可能会出现环，这样可以避免预测到之前的工具
                last_match_pos = match_positions[-1]
                matched_paths.append((path, last_match_pos + len(current_sequence)))
                total_frequency += path.frequency
        for idx in path_indices:
            path = self.paths[idx]
            path.confidence = path.frequency / total_frequency
        # 按路径置信度排序
        matched_paths.sort(key=lambda x: x[0].confidence, reverse=True)
        return matched_paths
    
    def _find_subsequence_positions(self, subseq: List[str], seq: List[str]) -> List[int]:
        """
        找出子序列在序列中的所有起始位置
        Args:
            subseq: 子序列
            seq: 序列
        Returns:
            子序列在序列中的起始位置列表
        """
        positions = []
        subseq_len = len(subseq)
        seq_len = len(seq)
        
        if subseq_len > seq_len:
            return positions
        
        for i in range(seq_len - subseq_len + 1):
            if seq[i:i+subseq_len] == subseq:
                positions.append(i)
        
        return positions
    
    def predict_inertial_path(self, current_sequence: List[str], max_steps: int = 3) -> List[str]:
        """
        预测惯性路径，基于当前工具调用序列预测后续可能的工具调用
        Args:
            current_sequence: 当前工具调用序列
            max_steps: 最大预测步数
        Returns:
            预测的后续工具调用序列
        """
        if not current_sequence:
            return []
        
        # 匹配当前序列与历史路径
        matched_paths = self.match_current_sequence(current_sequence)
        
        if not matched_paths:
            # 如果没有匹配的路径，使用边关系预测
            return self._predict_by_edges(current_sequence, max_steps)
        
        # 使用匹配度最高的路径进行预测
        best_path, match_pos = matched_paths[0]
        
        # 如果已经到达路径末尾，无法预测
        if match_pos >= len(best_path.tools):
            return []
        
        # 预测后续步骤
        subsequent_steps = best_path.tools[match_pos:match_pos+max_steps]
        return subsequent_steps
    
    def generate_essential_memory(self, current_tool_sequence: list):
        """根据工具调用序列生成essential_memory"""
        # only put the all target tool description into essential memory
        if current_tool_sequence[-1] not in self.nodes:
            return 0, ""
        essential_tools = ""
        tool_chain = self.predict_inertial_path(current_tool_sequence)
        previous_tools = self.get_previous_tools(current_tool_sequence[-1])
        next_tools = self.get_next_tools(current_tool_sequence[-1])
        # 合并
        # all_tools = previous_tools + next_tools + tool_chain
        all_tools = []
        for tool_name in tool_chain:
            all_tools.append(tool_name)
        for tool_name in previous_tools:
            all_tools.append(tool_name[0])
        for tool_name in next_tools:
            all_tools.append(tool_name[0])
        # 去重
        all_tools = list(set(all_tools))
        print(f'debug len of all_tools: {len(all_tools)}')
        # all_tools.sort(key=lambda x: x[1], reverse=True)
        # print(f'current_tool: {current_tool}')
        # print(f'previous_tools: {previous_tools}')
        # print(f'next_tools: {next_tools}')
        tool_num = 0
        for tool_name in all_tools:
            print(f'in data struct : generated tool_name: {tool_name}')
            if tool_name == 'GetTime':
                continue
            tool_num += 1
            essential_tools += f'{tool_num+1}. {self.nodes[tool_name].format_tool_description()}\n'
        essential_tools = ESSENTIAL_MEMORY_PREFIX.format(essential_tools=essential_tools)
        print(f'create {tool_num} essential tools memory')
        # print(f'essential_memory: {essential_memory}')
        return tool_num, essential_tools

    def update_previous_progress(self, previous_progress: str) -> None:
        """更新previous_progress"""
        self.previous_progress = previous_progress

    def update_task_description(self, task_description: str) -> None:
        """更新task_description"""
        self.task_description = task_description

    def _predict_by_edges(self, current_sequence: List[str], max_steps: int) -> List[str]:
        """
        基于边关系预测后续工具调用
        Args:
            current_sequence: 当前工具调用序列
            max_steps: 最大预测步数
        Returns:
            预测的后续工具调用序列
        """
        if not current_sequence:
            return []
        
        predicted_path = []
        current_tool = current_sequence[-1]
        
        for _ in range(max_steps):
            next_tools = self.get_next_tools(current_tool)
            if not next_tools:
                break
            
            # 选择调用次数最多的下一个工具
            next_tool, _ = next_tools[0]
            predicted_path.append(next_tool)
            current_tool = next_tool
        
        return predicted_path
    
    def find_common_patterns(self, min_length: int = 2, min_frequency: int = 2) -> List[Tuple[List[str], int]]:
        """
        发现常见的工具调用模式
        
        Args:
            min_length: 模式的最小长度
            min_frequency: 模式的最小出现频率
            
        Returns:
            常见模式及其频率列表 [(pattern, frequency)]
        """
        patterns = {}  # {pattern_tuple: frequency}
        
        # 从所有路径中提取子序列
        for path in self.paths:
            tools = path.tools
            path_len = len(tools)
            
            # 提取所有可能的子序列
            for length in range(min_length, path_len + 1):
                for i in range(path_len - length + 1):
                    pattern = tuple(tools[i:i+length])
                    patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # 过滤出频率达到阈值的模式
        common_patterns = []
        for pattern, frequency in patterns.items():
            if frequency >= min_frequency:
                common_patterns.append((list(pattern), frequency))
        
        # 按频率降序排序
        common_patterns.sort(key=lambda x: x[1], reverse=True)
        return common_patterns
    
    def analyze_param_dependencies(self, source_tool: str, target_tool: str) -> Dict[str, List[str]]:
        """分析两个工具之间的参数依赖关系"""
        pass
        
    
    def analyze_multi_step_dependencies(self, tool_sequence: List[str]) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
        """
        分析多步工具调用中的参数依赖关系
        Args:
            tool_sequence: 工具调用序列
        Returns:
            多步参数依赖关系 {target_tool: {param_name: [(source_tool, source_field)]}}
        """
        pass
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """获取工具的详细信息"""
        if tool_name not in self.nodes:
            return {}
        
        node = self.nodes[tool_name]
        return {
            "name": node.name,
            "description": node.description,
            "input_params": node.input_params,
            "output_params": node.output_params
        }
    
    def to_json(self) -> str:
        """将图结构转换为JSON字符串"""
        graph_data = {
            "nodes": {},
            "edges": [],
            "paths": []
        }
        
        # 添加节点信息
        for tool_name, node in self.nodes.items():
            graph_data["nodes"][tool_name] = {
                "name": node.name,
                "description": node.description,
                "input_params": node.input_params
            }
        
        # 添加边信息
        for source, targets in self.edges.items():
            for target, edge in targets.items():
                graph_data["edges"].append({
                    "source": source,
                    "target": target,
                    "call_count": edge.call_count,
                    "param_mappings": edge.param_mappings
                })
        
        # 添加路径信息
        for path in self.paths:
            graph_data["paths"].append({
                "tools": path.tools,
                "frequency": path.frequency
            })
        
        return json.dumps(graph_data, indent=2, ensure_ascii=False)
    
    def get_param_sources(self, matched_tool_sequence, target_tool, miss_param_name):
        """
        查找目标工具缺失参数的惯性来源 (基于 param_edges)
        
        Args:
            matched_tool_sequence: 当前匹配到的工具序列 (当前版本未使用)
            target_tool: 目标工具名称
            miss_param_name: 缺失的参数名称
            
        Returns:
            List[Tuple[str, str, int]]: 可能的惯性来源列表，每个元素为 (源工具名, 源参数名, 依赖计数)，按依赖计数降序排序
        """
        results = []
        
        # 检查目标工具和参数是否存在于 param_edges 结构中
        if target_tool not in self.param_edges or miss_param_name not in self.param_edges[target_tool]:
            return results # 没有关于这个目标参数的记录
            
        # 获取所有指向该目标参数的 ParamEdge 对象
        param_source_edges = self.param_edges[target_tool][miss_param_name]
        
        # 收集来源信息
        for (source_tool, source_param), param_edge in param_source_edges.items():
            results.append((source_tool, source_param, param_edge.count))
        
        # 按依赖计数降序排序
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def attach_param_edge(self, history, tool_name, inputs):
        """
        将参数边附加到图的 param_edges 属性中，记录参数依赖关系
        
        Args:
            history: 执行历史记录
            tool_name: 当前工具名称 (target_tool)
            inputs: 工具输入参数
        """
        if tool_name not in self.nodes:
            print(f"警告: 目标工具 '{tool_name}' 不在图中，无法附加参数边。")
            return
            
        if not inputs:
            return # 没有输入参数，无需处理
            
        # 获取工具的参数图，用于添加示例值
        tool_node = self.nodes[tool_name]
        param_graph = tool_node.param_graph
        
        # 逆向遍历历史记录，寻找参数来源
        for target_param_name, param_value in inputs.items():
            if param_value is None: # 跳过空值
                continue
                
            # 添加参数值示例到参数节点 (如果参数存在于描述中)
            if target_param_name in param_graph.nodes:
                param_graph.nodes[target_param_name].add_value_example(param_value)
            
            # 寻找匹配的历史输出
            found_match_in_history = False
            for record in reversed(history.history):
                source_tool_name = record.get("tool_name")
                outputs = record.get("outputs", {}) # outputs 是一个字典
                
                # 跳过当前工具自身记录、未知工具或没有输出的记录
                if source_tool_name == tool_name or source_tool_name == "unknown" or not outputs:
                    continue
                
                # 检查源工具是否存在于图中 (虽然可能不是必须的，但保持一致性)
                if source_tool_name not in self.nodes:
                    # print(f"警告: 源工具 '{source_tool_name}' 不在图中，但仍会尝试匹配参数。")
                    pass # 允许来自未知工具的参数匹配，只要历史记录中有
                    
                # 检查历史输出中是否能匹配当前输入参数值
                matched_source_param = None
                for source_param_name, output_value in outputs.items():
                    # 各种匹配逻辑
                    match_found = False
                    if output_value == param_value: match_found = True
                    elif isinstance(output_value, list) and param_value in output_value: match_found = True
                    elif isinstance(output_value, (dict, set)):
                        if isinstance(output_value, dict):
                            if param_value in output_value or param_value in output_value.values(): match_found = True
                        elif isinstance(output_value, set) and param_value in output_value: match_found = True

                    if match_found:
                        matched_source_param = source_param_name
                        break # 找到一个匹配就停止内层循环
                        
                # 如果找到匹配的源参数
                if matched_source_param is not None:
                    param_edge_key = (source_tool_name, matched_source_param)
                    target_param_dict = self.param_edges[tool_name][target_param_name]
                    print(f'[DEBUG INFO]: attach_param_edge: {param_edge_key} -> ({tool_name}, {target_param_name})')
                    # 检查 ParamEdge 是否已存在
                    if param_edge_key in target_param_dict:
                        # 已存在，增加计数
                        print(f"  更新 ParamEdge: ({source_tool_name}, {matched_source_param}) -> ({tool_name}, {target_param_name}), count={target_param_dict[param_edge_key].count}")
                        target_param_dict[param_edge_key].increment_count()
                        # print(f"  更新 ParamEdge: ({source_tool_name}, {matched_source_param}) -> ({tool_name}, {target_param_name}), count={target_param_dict[param_edge_key].count}")
                    else:
                        print(f"  创建新的 ParamEdge: ({source_tool_name}, {matched_source_param}) -> ({tool_name}, {target_param_name}), count=1")
                        # 不存在，创建新的 ParamEdge
                        new_param_edge = ParamEdge(source_tool_name, matched_source_param, tool_name, target_param_name)
                        target_param_dict[param_edge_key] = new_param_edge
                        # print(f"  创建 ParamEdge: ({source_tool_name}, {matched_source_param}) -> ({tool_name}, {target_param_name}), count=1")
                        
                    found_match_in_history = True
                    break # 找到当前参数的来源，停止对更早历史的搜索
                    
            # (可选) 如果遍历完历史都没找到匹配，可以打印信息
            # if not found_match_in_history:
            #     print(f"  未在历史记录中找到参数 '{target_param_name}' (值: {param_value}) 的来源。")

    def update_graph(self, current_sequence: Dict) -> None:
        # print(f'[DEBUG INFO]: call update_graph to update tool graph')
        """
        从current_sequence更新图结构，支持根据惯性步骤进行序列分割
        Args:
            current_sequence: 当前序列数据，包含steps和inertial_step等信息
        """
        if not current_sequence:
            print("Warning: Empty current_sequence provided to update_graph")
            return
        history = ExecutionHistory(max_len=40)
        # --- 预处理 ---
        steps = current_sequence.get("steps", [])
        # print(f"[DEBUG] steps: {steps}")
        # print(f"[DEBUG INFO]: current_sequence: {steps}")
        if not steps:
            print("Warning: No steps found in current_sequence")
            return
            
        # 提取惯性步骤信息，格式为 [(task_id, step_index, action), ...]
        inertial_steps = current_sequence.get("inertial_step", [])
        # print(f"[DEBUG INFO]: inertial_steps: {inertial_steps}")
        inertial_indices = set()
        
        for inertial_step in inertial_steps:
            inertial_indices.add(inertial_step[0])  
        
        # print(f"Updating graph with trajectory containing {len(steps)} steps, {len(inertial_indices) + 1} chain")
        
        # 初始化工具序列和分割点
        tool_sequences = []
        current_sequence_tools = []
        
        # --- 分割 ---
        for i, step in enumerate(steps):
            # print(f"[DEBUG] step {i}: {step}")
            if step.get("type") == "initial":
                continue
            if i in inertial_indices:
                if current_sequence_tools:  # 只在有内容时 append
                    tool_sequences.append(current_sequence_tools)
                current_sequence_tools = []
                continue

            if step.get("type") == "act_ob_pair":
                action_data = step.get("action", {})
                observation_data = step.get("observation", {})
                parsed_observation = observation_data.get("parsed_content", None)
                parsed_action = action_data.get("parsed_content", None)
                
                # print(f"[DEBUG] parsed_content: {parsed_action}")
                tool_name = None
                if parsed_action and isinstance(parsed_action, dict) and parsed_action.get("tool_name") not in ["unknown", None]:
                    tool_name = parsed_action.get("tool_name")
                    # print(f'[DEBUG INFO]: tool_name: {tool_name}')
                    current_sequence_tools.append(tool_name)
                    self.attach_param_edge(history, tool_name, parsed_action.get("inputs", {}))
                    record = {
                        "tool_name": tool_name,
                        "inputs": parsed_action.get("inputs", {}),
                        "outputs": parsed_observation,
                    }
                    history.add_record(record)

                
                # else:  # 不要在这里分割
                #     if current_sequence_tools:
                #         tool_sequences.append(current_sequence_tools)
                #     current_sequence_tools = []
        # 循环结束后，补充最后一段
        if current_sequence_tools:
            tool_sequences.append(current_sequence_tools)
        # --- 更新 --- 
        total_tools = 0
        print(f"Identified {len(tool_sequences)} sub-sequences after splitting at inertial steps")

        # print(f'*' * 50)
        # print(f"Updating graph with {tool_sequences}")
        # print(f'*' * 50)
        for idx, sequence in enumerate(tool_sequences):
            # print(f"  - Processing sub-sequence {idx+1}: {sequence}")
            if len(sequence) > 1:  # 忽略长度为1的序列（无调用关系）
                if self.debug: print(f"  - Updating graph with sub-sequence {idx+1}: {sequence}")
                try:
                    self.record_tool_sequence(sequence)
                    total_tools += len(sequence)
                except Exception as e:
                    print(f"Error updating sub-sequence {idx+1}: {e}")
            else:
                print(f"  - Skipping sub-sequence {idx+1} with length {len(sequence)}: {sequence}")
        
        # 打印统计信息
        print(f"Graph update complete: {len(self.paths)} paths, {total_tools} tools")
        print(f"Path lookup table size: {len(self.chain2ID)}")

    def load_from_json(self, tool_path: str, chain_path: str) -> None:
        """从JSON字符串加载图结构"""
        self.load_tool_description_from_json(tool_path)
        self.load_tool_chain_from_json(chain_path)
        print(f"Tool Graph loaded from {tool_path} and {chain_path}")
    
    def load_tool_chain_from_json(self, chain_path: str) -> None:
        if os.path.isabs(chain_path) is False:
            chain_path = os.path.join(os.path.dirname(__file__), chain_path)
        try:
            if os.path.getsize(chain_path) == 0:
                print(f"从空轨迹开始，不加载任何轨迹")
                return
            else: 
                with open(chain_path, "r", encoding="utf-8") as f:
                    tool_chains = json.load(f)
                    tool_chains = tool_chains.get("sequences", {})
                    print(f"预计加载 {len(tool_chains)} 个工具调用序列")
            
            # 加载边和路径信息 - 从新的日志格式解析
            for trajectory in tool_chains:
                tool_sequence = []
                steps = trajectory.get("steps", [])
                for step in steps:
                    if step.get("type") == "act_ob_pair":
                        action = step.get("action", {})
                        parsed_content = action.get("parsed_content", {})
                        tool_name = parsed_content.get("tool_name")
                        # 确保提取到了有效的工具名称
                        if tool_name and tool_name != "unknown": # 可以根据需要调整过滤条件
                            tool_sequence.append(tool_name)

                if tool_sequence: # 确保序列不为空
                    # 使用 record_tool_sequence 同时更新边和路径
                    self.record_tool_sequence(tool_sequence)

            print(f"加载了 {len(self.paths)} 条路径")
            print(f"路径查询表大小: {len(self.chain2ID)}")
            
        except Exception as e:
            print(f"从JSON加载边失败: {e}")
            import traceback
            traceback.print_exc()
        pass

    def load_tool_description_from_json(self, tool_path: str) -> None:
        if os.path.isabs(tool_path) is False:
            tool_path = os.path.join(os.path.dirname(__file__), tool_path)
        
        try:
            with open(tool_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
                print(f"预计加载 {len(graph_data)} 个工具")
            print(f'DEBUG_INFO: load tool description from {tool_path}')

            # 加载节点信息
            for tool_name, node_data in graph_data.items():
                # 构造简化的工具描述
                tool_desc = {
                    "tool": tool_name,
                    "tool_desc": node_data.get("tool_desc", ""),
                    "args": [],
                    "returns": []
                }
                args = node_data.get("args", [])
                returns = node_data.get("returns", [])
                if args:
                    content = args
                else:
                    content = {}
                for param in content:
                    tool_desc["args"].append(param)
                if returns:
                    content = returns
                else:
                    content = {}
                for param in content:
                    tool_desc["returns"].append(param)

                self.add_node(tool_name, tool_desc)

            print(f"加载了 {len(self.nodes)} 个工具")
            
        except Exception as e:
            print(f"从JSON加载结点失败: {e}")
            import traceback
            traceback.print_exc()

########################################################
# 基于工具链语义相似度预测下一个工具, appended on 3.24 by jjy
########################################################

    def _get_path_text_representation(self, path: List[str], end_pos: int, ROI_len: int = 1) -> str:
        """获取工具路径的文本表示，优化语义表示方式"""
        path_texts = []
        # print(f"DEBUG: path: {path}, end_pos: {end_pos}, ROI_len: {ROI_len}")
        # 收集工具信息
        tools_info = []
        for tool_name in path[end_pos:end_pos+ROI_len]:
            if tool_name in self.nodes:
                tools_info.append({
                    "name": tool_name,
                    "desc": self.nodes[tool_name].description
                })
        
        # 构建更丰富的路径表示
        if len(tools_info) > 0:
            # 添加整体路径概述
            # path_texts.append(f"工具调用链: {' -> '.join([t['name'] for t in tools_info])}")
            
            # 添加详细工具信息
            for i, tool in enumerate(tools_info):
                # 序号前缀，帮助模型理解调用顺序
                prefix = f"Step{i+1}: "
                
                # 获取工具的详细信息
                tool_desc = tool['desc'] if tool['desc'] else "no description"
                # 构建该工具的文本表示
                tool_text = f"{prefix}use {tool['name']}({tool_desc})"
                path_texts.append(tool_text)
            
            # 如果路径长度大于1，添加功能流描述
            if len(tools_info) > 1:
                path_texts.append(f"该调用链完成了从{tools_info[0]['name']}到{tools_info[-1]['name']}的操作流")
        # print(f"DEBUG: in get_path_text on path_texts: {path_texts}")
        # 将所有文本连接起来
        return "\n".join(path_texts)


    def _get_confidence(self, x) -> float:
        """计算路径的置信度"""
        # f = 1 - pow(1.1, -x)  # 使用指数衰减函数 alfworld
        f = 1 - pow(1.1, -x) 
        return f

    def predict_next_tool_with_chain_similarity(self, current_sequence: List[str], intuition: str ="", thereshold=0.4, alpha=0.5):
        """
        通过计算整个工具链的语义相似度预测下一个工具
        Args:
            current_sequence: 当前工具调用序列
            thereshold: 阈值，默认0.6
            alpha: 频率得分权重，默认0.4
        Returns:
            是否存在惯性调用，如果存在则返回True，否则返回False
            预测的下一个工具名称，如果无法预测则返回空字符串
        """
        if not current_sequence:
            return False, ""
        
        # print(f"当前调用序列: {current_sequence}")
        
        # 1. 获取包含当前工具调用链作为子集的所有工具调用路径
        matching_paths = []
        
        # 使用字典记录已添加的路径，避免重复
        processed_paths = {}

        # --- 下一步预测的工具 ---
        predicted_tool = []
        # 找到父路径（使用副本而非直接引用）
        if current_sequence[0] in self.tooll2chainID:
            potential_path_indices = self.tooll2chainID[current_sequence[0]].copy()  # 使用copy()创建副本
        else:
            potential_path_indices = set()
        
        for i in range(1, len(current_sequence)):
            if current_sequence[i] not in self.tooll2chainID:
                potential_path_indices = set()  # 如果序列中任何工具没有路径索引，结果就是空集
                break
            t = self.tooll2chainID[current_sequence[i]]
            potential_path_indices &= t  # 使用交集运算符而非原地修改
        
        candidate_tool_dic = defaultdict(float)
        total_frequency = 0
        for path_idx in potential_path_indices:
            # 检查索引是否有效
            if path_idx >= len(self.paths):
                print(f"警告: 路径索引 {path_idx} 超出范围，最大索引为 {len(self.paths)-1}")
                continue
            # print('debug on list index out of r: ', path_idx)
            # print(f'max self.paths: {len(self.paths)-1}')
            path = self.paths[path_idx]
            is_subseq, start_pos = is_subsequence(current_sequence, path.tools)
            if is_subseq and start_pos + len(current_sequence) < len(path.tools):
                # 确保索引在有效范围内
                end_pos = start_pos + len(current_sequence)
                next_tool = path.tools[end_pos]
                candidate_tool_dic[next_tool] += path.frequency
                total_frequency += path.frequency
        
        # print(f'[DEBUG] candidate_tool_dic: {candidate_tool_dic}')
         # 2. 计算每个候选工具链的得分
        next_tool_scores = {}
        
        # 2.1 基于频率的得分
        # total_frequency = sum(frequency for _, frequency in candidate_tool_dic.items())
        print(f'[DEBUG] total_frequency: {total_frequency}')
        for tool_name, frequency in candidate_tool_dic.items():
            if tool_name not in next_tool_scores:
                next_tool_scores[tool_name] = {
                    "frequency_score": 0,
                    "semantic_score": 0,
                    "combined_score": 0
                }
            # 基础频率得分
            next_tool_scores[tool_name]["frequency_score"] = (frequency / total_frequency) * self._get_confidence(total_frequency)
        
        start_time = time.time()
        if intuition is None:
            intuition = self.task_description
        # 2.2 如果提供了任务描述，计算整个工具链的语义相似度
        if intuition and intuition.strip():            
            # 获取任务嵌入
            # print(f" intuition 嵌入")
            intuition_embedding = get_embedding(intuition)
            task_end_time = time.time()
            # 计算每个候选工具链的语义相似度
            for tool_name, _ in candidate_tool_dic.items():
                # 获取整个工具链的文本表示
                tool_text = str(tool_name) + ": " + str(self.nodes[tool_name].description)
                # tool_text = self._get_path_text_representation(list(path_key), end_pos)

                print(f'[debug]: path_text: {tool_text}')
                # print(f'[debug]: intuition_text: {intuition}')
                path_embedding = get_embedding(tool_text)
                semantic_score = compute_similarity(path_embedding, intuition_embedding)
                next_tool_scores[tool_name]["semantic_score"] = semantic_score

        # 3. 综合评分并找出最佳路径
        best_path = None
        best_score = -1
        
        for tool_name, scores in next_tool_scores.items():
            # 综合频率和语义得分 (权重可调整)
            combined_score = alpha * scores["frequency_score"] + (1 - alpha) * scores["semantic_score"]
            scores["combined_score"] = combined_score  # 添加到得分字典中便于输出
            
        # 输出所有路径得分，便于分析
        sorted_path_scores = sorted(next_tool_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        if len(sorted_path_scores) == 0:
            print("没有找到匹配的路径")
            return False, ""
        best_tool_name = sorted_path_scores[0][0]
        best_score = sorted_path_scores[0][1]['combined_score']
        # 输出前三个路径的得分
        # for i, (path_key, scores) in enumerate(sorted_path_scores[:3]):
        #     print(f"路径 {i+1}, 综合得分: {scores['combined_score']:.4f}")
        # print(f'=' * 50)
        print(f"combined_score: {sorted_path_scores[0][1]['combined_score']}")
        print(f"frequency_score: {sorted_path_scores[0][1]['frequency_score']}")
        print(f"semantic_score: {sorted_path_scores[0][1]['semantic_score']}")
        if not best_tool_name:   
            print("无法确定最佳路径")
            return False, ""
        # 4. 检查是否存在明显的惯性调用场景 (得分高于阈值)
        if best_score > thereshold:  # 阈值设为0.6，可以根据实际情况调整
            print(f"检测到惯性调用，得分: {best_score:.4f}, 阈值: {thereshold}")
            return True, best_tool_name
        
        # 得分不够高，考虑其他因素
        print(f"得分不足以确定为惯性调用，得分: {best_score:.4f}, 阈值: {thereshold}, tool_name: {best_tool_name}")
        # return False, self._consider_chain_factors(current_sequence, path_scores)
        return False, ""
    

    def _predict_next_by_edges(self, current_tool: str) -> str:
        """基于边关系预测下一个工具"""
        if current_tool not in self.edges:
            print(f"[DEBUG INFO]: 当前工具 {current_tool} 不在图中")
            return ""
        # 获取所有可能的下一个工具
        next_tools = []
        for target, edge in self.edges[current_tool].items():
            next_tools.append((target, edge.call_count))
        
        if not next_tools:
            return ""
        
        # 按调用次数排序
        next_tools.sort(key=lambda x: x[1], reverse=True)
        
        # 返回调用次数最多的工具
        return next_tools[0][0]

    # def _consider_chain_factors(self, current_sequence: List[str], path_scores: Dict) -> str:
    #     """考虑额外因素优化工具选择"""
    #     # 提取候选工具及其得分
    #     if not path_scores:
    #         return self._predict_next_by_edges(current_sequence[-1])
        
    #     # 选择综合分数最高的路径的下一个工具
    #     sorted_path_scores = sorted(path_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
    #     if sorted_path_scores:
    #         best_path = sorted_path_scores[0][0]
    #         # print(f"选择综合分数最高的路径: 综合分数={path_scores[best_path]['combined_score']:.4f}")
    #         return path_scores[best_path]["next_tool"]
        
    #     # 如果没有有效的路径得分，回退到基于边关系的预测
    #     return self._predict_next_by_edges(current_sequence[-1])

    def predict_next_tool_with_edge_similarity(self, current_sequence: List[str]) -> Tuple[bool, str]:
        """
        专门基于边关系预测下一个工具
        
        Args:
            current_sequence: 当前工具调用序列
        Returns:
            是否存在明确的预测关系，预测的下一个工具名称
        """
        if not current_sequence:
            return False, ""
        
        current_tool = current_sequence[-1]
        next_tools = self.get_next_tools(current_tool)
        
        if not next_tools:
            return False, ""
        
        # 获取下一个工具及其调用次数
        next_tool, call_count = next_tools[0]
        
        # 如果调用次数超过阈值，认为是较强的边关系
        has_strong_edge = call_count > 3
        
        print(f"边关系预测: {current_tool} -> {next_tool} (调用次数: {call_count})")
        
        return has_strong_edge, next_tool

def is_subsequence(subseq: List[str], seq: List[str]) -> Tuple[bool, int]:
    # 将deque转换为列表以支持切片操作
    seq_list = list(seq)
    # print(f"DEBUG: is_subsequence: {subseq}, seq_list: {seq_list}")
    
    if len(subseq) > len(seq):
        return False, -1
    for i in range(len(seq) - len(subseq) + 1):
        match = True
        for j in range(len(subseq)):
            if seq_list[i + j] != subseq[j]:
                match = False
                break
        if match:
            return True, i
    return False, -1


# if __name__ == '__main__':
#     # 测试代码
#     tool_graph = ToolGraph()
#     tool_graph.load_tool_description_from_json('/data/FastToolCalling/src/AutoTool/graph/alfworld_action_description.json')
#     print(tool_graph.nodes["open"].input_params)
#     # file_path = '/data/agentboard/examples/trajectories/inertia_agent_calls_20250421_214838.json'
#     # with open(file_path, "r", encoding="utf-8") as f:
    #     current_sequence = json.load(f)
    #     print(f"Loaded JSON type: {type(current_sequence)}")
    #     tool_graph.update_graph(current_sequence.get("sequences")[0])

import matplotlib.pyplot as plt
import os
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch

def plot_successor_pie_chart(entity_name: any,
                             successors_map: dict,
                             total_frequency: int,
                             output_dir: str,
                             entity_type: str = "pair"
                             ):
    """
    Generates a final, publication-quality pie chart.
    - Legend hatches are white with a black border.
    - All slices >= 10% receive a distinct hatch pattern.
    - Color and hatch rules are applied consistently.
    """
    # --- 1. Style Setup ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['hatch.linewidth'] = 1.5
    # --- 核心修改: Set the GLOBAL default hatch color to white ---
    plt.rcParams['hatch.color'] = 'white'

    # --- 2. Data Preparation ---
    # ... (Filtering logic remains the same) ...
    filtered_successors_map = {}
    if entity_type == "pair":
        tool_B = entity_name[1]
        filtered_successors_map = {succ: freq for succ, freq in successors_map.items() if succ != tool_B}
    elif entity_type == "single":
        tool_A = entity_name
        filtered_successors_map = {succ: freq for succ, freq in successors_map.items() if succ != tool_A}
    
    total_frequency = sum(filtered_successors_map.values())
    if total_frequency == 0: return

    sorted_successors = sorted(filtered_successors_map.items(), key=lambda item: item[1], reverse=True)
    labels = [succ.replace("_", " ") for succ, freq in sorted_successors]
    sizes = [freq for succ, freq in sorted_successors]
    percentages = [(s / total_frequency) * 100 if total_frequency > 0 else 0 for s in sizes]
    if not sizes: return

    # --- 3. Dynamic Color Assignment ---
    # ... (Color assignment logic remains the same) ...
    light_orange_color = '#FDB562'
    blue_color = 'tab:blue'
    green_color = 'tab:green'
    special_colors = ['tab:orange', 'tab:blue', 'tab:green']
    other_colors_pool = [c for c in plt.get_cmap('tab10').colors if c not in special_colors]
    
    final_colors = []
    for i in range(len(sizes)):
        if i == 0: final_colors.append(light_orange_color)
        elif i == 1: final_colors.append(blue_color)
        elif i == 2: final_colors.append(green_color)
        else: final_colors.append(other_colors_pool[(i - 3) % len(other_colors_pool)])

    # --- 4. 核心修改: Dynamic Hatch Assignment for ALL Slices >= 10% ---
    hatches_ordered = ['xx', '/', '\\', 'o', 'O', '.', '*', '+', '|']
    assigned_hatches = []
    hatch_idx = 0
    for i in range(len(sizes)):
        if percentages[i] >= 10:
            assigned_hatches.append(hatches_ordered[hatch_idx % len(hatches_ordered)])
            hatch_idx += 1
        else:
            assigned_hatches.append('') # No hatch for small slices

    # --- 5. Create Figure and Plot ---
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(top=0.8)

    def autopct_conditional(pct):
        return f'{pct:.1f}%' if pct >= 10 else ''

    wedges, texts, autotexts = ax.pie(
        sizes,
        autopct=autopct_conditional,
        startangle=90,
        pctdistance=0.85,
        explode=[0.02] * len(sizes),
        colors=final_colors
    )
    
    # --- 6. Styling Loop ---
    for i, wedge in enumerate(wedges):
        # Apply the dynamically assigned hatch
        wedge.set_hatch(assigned_hatches[i])
        wedge.set_edgecolor('white')
        wedge.set_linewidth(1.5)

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(32)
        autotext.set_fontweight('bold')

    ax.axis('equal')

    # --- 7. Centered Title and Filtered Legend ---
    if entity_type == "pair":
        title_str = f"{entity_name[0].replace('_', ' ')} → {entity_name[1].replace('_', ' ')} ({total_frequency})"
    else:
        title_str = f"{entity_name.replace('_', ' ')} ({total_frequency})"
    fig.suptitle(title_str, fontsize=40, fontweight='bold', y=0.97)
    
    legend_elements = []
    for i, label in enumerate(labels):
        if percentages[i] >= 10:
            legend_patch = Patch(
                facecolor=final_colors[i],
                # 核心修改: edgecolor is for the border, hatch color is now globally white
                edgecolor='black', 
                hatch=assigned_hatches[i],
                label=label
            )
            legend_elements.append(legend_patch)

    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=min(len(legend_elements), 5),
        frameon=True,
        edgecolor='black',
        fontsize=22
    )
    
    # --- 8. Save Figure ---
    if entity_type == "pair":
        filename_prefix = f"successors_of_{entity_name[0]}_then_{entity_name[1]}"
    else:
        filename_prefix = f"successors_of_{entity_name}"
    
    filename = f"{filename_prefix}_final_v3.png".replace(" ", "_").replace("→", "to")
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"    Final styled pie chart (v3) saved to: {filepath}")
    plt.close(fig)

    # Reset rcParams if you have other plots in the same script
    plt.rcdefaults()

def generate_parameter_inertia_table(tool_graph, target_tool_name: str):
    """
    Generates and prints a table showing the sources of input parameters for a given target tool.
    """
    if target_tool_name not in tool_graph.nodes:
        print(f"  Error: Target tool '{target_tool_name}' not found in the tool graph. Cannot generate parameter inertia table.")
        return

    target_node = tool_graph.nodes[target_tool_name]
    target_input_params = list(target_node.input_params.keys()) 

    if not target_input_params:
        print(f"  Tool '{target_tool_name}' has no defined input parameters.")
        return

    print(f"  Parameter Inertia for Tool: {target_tool_name}")
    print("  ------------------------------------------------------------------------------------")
    print(f"  {'Target Param':<20} | {'Source Tool':<20} | {'Source Param (Output)':<25} | {'Frequency':<10} | {'Proportion':<10}")
    print("  ------------------------------------------------------------------------------------")

    found_any_dependency = False

    if target_tool_name not in tool_graph.param_edges:
        print(f"  No parameter dependency edges recorded for tool '{target_tool_name}'.")
        for target_param_name in target_input_params:
             print(f"  {target_param_name:<20} | {'N/A':<20} | {'N/A':<25} | {'0':<10} | {'0.0%':<10}")
        print("  ------------------------------------------------------------------------------------")
        return

    param_dependencies_for_target_tool = tool_graph.param_edges[target_tool_name]

    for target_param_name in target_input_params:
        if target_param_name in param_dependencies_for_target_tool:
            sources_for_this_param = param_dependencies_for_target_tool[target_param_name]
            
            if not sources_for_this_param:
                print(f"  {target_param_name:<20} | {'No specific sources':<20} | {'N/A':<25} | {'-':<10} | {'-':<10}")
                continue

            total_frequency_for_this_param = sum(edge.count for edge in sources_for_this_param.values())
            
            if total_frequency_for_this_param == 0: 
                 print(f"  {target_param_name:<20} | {'Sources found but total freq is 0':<20} | {'N/A':<25} | {'-':<10} | {'-':<10}")
                 continue

            sorted_sources = sorted(sources_for_this_param.items(), key=lambda item: item[1].count, reverse=True)

            first_source = True
            for (source_tool, source_param_name), param_edge_obj in sorted_sources:
                found_any_dependency = True
                proportion = (param_edge_obj.count / total_frequency_for_this_param) * 100 if total_frequency_for_this_param > 0 else 0
                
                param_display_name = target_param_name if first_source else "" 
                print(f"  {param_display_name:<20} | {source_tool:<20} | {source_param_name:<25} | {param_edge_obj.count:<10} | {proportion:>9.1f}%")
                first_source = False
            
            if not first_source : 
                if len(sources_for_this_param) >1 : print(f"  {'':<20} | {'-'*20} | {'-'*25} | {'-'*10} | {'-'*10}")

        else: 
            print(f"  {target_param_name:<20} | {'No recorded sources':<20} | {'N/A':<25} | {'0':<10} | {'0.0%':<10}")
    
    if not found_any_dependency and any(target_param_name not in param_dependencies_for_target_tool for target_param_name in target_input_params):
        pass 

    print("  ------------------------------------------------------------------------------------") 

def main(tool_description_path: str, 
         tool_trajectory_path: str, 
         high_freq_tool_edge_threshold: int = 5,
         tool_pairs_for_pie_chart: list = None,
         tools_for_parameter_inertia_table: list = None, # <--- 修改这里，表示接收一个列表
         output_pie_charts_dir: str = "tool_successor_pie_charts",
         tools_for_single_successor_pie_chart: list = None
         ):
    # 1. 初始化 ToolGraph
    tool_graph = ToolGraph()
    tool_graph.debug = False # Set to True for more verbose output from ToolGraph methods

    # 2. 加载工具描述
    print(f"--- Loading Tool Descriptions from: {tool_description_path} ---")
    if not os.path.exists(tool_description_path):
        print(f"Error: Tool description file not found at {tool_description_path}")
        return
    tool_graph.load_tool_description_from_json(tool_description_path)
    if not tool_graph.nodes:
        print("Error: No tool descriptions loaded. Check the file format and content. Exiting.")
        return
    print(f"Successfully loaded {len(tool_graph.nodes)} tool descriptions.")

    # 3. 加载工具轨迹并更新图
    print(f"\n--- Loading Tool Trajectories from: {tool_trajectory_path} ---")
    if not os.path.exists(tool_trajectory_path):
        print(f"Error: Tool trajectory file not found at {tool_trajectory_path}")
        # Continue without trajectories if you want to analyze just the tool descriptions
        # or handle this case as an error. For now, we'll return.
        return
        
    try:
        with open(tool_trajectory_path, "r", encoding="utf-8") as f:
            trajectory_data = json.load(f)
        
        # The structure of trajectory_data can vary.
        # Based on datastruct.py, update_graph expects a single sequence dictionary.
        # If your file contains a list of sequences under a "sequences" key:
        sequences_list = trajectory_data.get("sequences")
        
        if isinstance(sequences_list, list):
            print(f"Found {len(sequences_list)} sequences to process.")
            if not sequences_list:
                 print(f"Warning: 'sequences' list is empty in {tool_trajectory_path}")
            for i, seq_data_item in enumerate(sequences_list):
                if not isinstance(seq_data_item, dict):
                    print(f"Warning: Sequence item {i} is not a dictionary, skipping.")
                    continue
                print(f"\nProcessing sequence {i+1}/{len(sequences_list)}...")
                tool_graph.update_graph(seq_data_item)
        elif isinstance(trajectory_data, dict) and "steps" in trajectory_data: # If the file itself is a single sequence
            print("Processing the trajectory file as a single sequence...")
            tool_graph.update_graph(trajectory_data)
        else:
            print(f"Warning: Could not find a list of sequences under 'sequences' key, nor a single sequence dict in {tool_trajectory_path}. Check file structure.")
            print("Attempting to process file as a list of sequences if it's a list directly.")
            if isinstance(trajectory_data, list):
                for i, seq_data_item in enumerate(trajectory_data):
                    if not isinstance(seq_data_item, dict):
                        print(f"Warning: Sequence item {i} is not a dictionary, skipping.")
                        continue
                    print(f"\nProcessing sequence {i+1}/{len(trajectory_data)}...")
                    tool_graph.update_graph(seq_data_item)


    except Exception as e:
        print(f"Error loading or processing trajectories from {tool_trajectory_path}: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 打印 ToolGraph 的 to_json() 输出 (可选，可能非常长)
    print(f"\n--- ToolGraph JSON Representation (Summary) ---")
    # For brevity, let's print counts instead of the full JSON if it's too large
    print(f"  Number of tools (nodes): {len(tool_graph.nodes)}")
    edge_count = sum(len(targets) for targets in tool_graph.edges.values())
    print(f"  Number of tool call edges: {edge_count}")
    print(f"  Number of unique tool paths recorded: {len(tool_graph.paths)}")
    param_edge_count = 0
    for target_tool_data in tool_graph.param_edges.values():
        for source_map in target_tool_data.values():
            param_edge_count += len(source_map)
    print(f"  Number of unique parameter dependency edges: {param_edge_count}")
    # To print the full JSON (can be very large):
    # full_json_output = tool_graph.to_json()
    # with open("tool_graph_output.json", "w", encoding="utf-8") as f_out:
    #     f_out.write(full_json_output)
    # print("Full ToolGraph JSON representation saved to tool_graph_output.json")

    # Print N most frequent tool paths
    print(f"\n--- Most Frequent Tool Paths ---")
    num_top_paths_to_print = 10 # Or make this a parameter to main()
    if not tool_graph.paths:
        print("  No tool paths recorded in the graph.")
    else:
        # Sort paths by frequency in descending order
        sorted_paths = sorted(tool_graph.paths, key=lambda p: p.frequency, reverse=True)
        print(f"  Top {min(num_top_paths_to_print, len(sorted_paths))} most frequent paths (out of {len(sorted_paths)} unique paths):")
        for i, path_obj in enumerate(sorted_paths[:num_top_paths_to_print]):
            path_str = " -> ".join(path_obj.tools)
            print(f"    {i+1}. Path: [{path_str}], Frequency: {path_obj.frequency}")

    # Analyze successors for a specific tool pair relationship
    print(f"\n--- Successor Analysis for Specific Tool Pair ---")
    # Define the tool pair: (source_tool_of_interest, then_analyze_successors_of_this_tool)
    # For example, if we want to see if "go_to" -> "look_around" exists, 
    # and if so, what follows "look_around".
    source_tool_of_interest = "go_to"  # Change as needed
    target_whose_successors_to_analyze = "look_around" # Change as needed

    print(f"  Analyzing if '{source_tool_of_interest}' -> '{target_whose_successors_to_analyze}' exists, and then successors of '{target_whose_successors_to_analyze}'.")

    # Check if source_tool_of_interest exists
    if source_tool_of_interest not in tool_graph.nodes:
        print(f"    Source tool '{source_tool_of_interest}' not found in the graph.")
    # Check if source_tool_of_interest has outgoing edges and if target_whose_successors_to_analyze is among them
    elif source_tool_of_interest not in tool_graph.edges or \
         target_whose_successors_to_analyze not in tool_graph.edges[source_tool_of_interest]:
        print(f"    No direct call edge found from '{source_tool_of_interest}' to '{target_whose_successors_to_analyze}'.")
    else:
        # The A -> B relationship exists, print its frequency
        call_count_A_to_B = tool_graph.edges[source_tool_of_interest][target_whose_successors_to_analyze].call_count
        print(f"    Confirmed: '{source_tool_of_interest}' -> '{target_whose_successors_to_analyze}' exists with {call_count_A_to_B} calls.")

        # Now, analyze successors of target_whose_successors_to_analyze (Tool B)
        print(f"\n    Analyzing successors of '{target_whose_successors_to_analyze}':")
        tool_B_name = target_whose_successors_to_analyze
        if tool_B_name not in tool_graph.nodes:
            # This case should be rare if the A->B edge exists, but good to check
            print(f"      Tool '{tool_B_name}' (the target of the pair) not found as a node in the graph, though an edge to it exists.")
        elif tool_B_name in tool_graph.edges and tool_graph.edges[tool_B_name]:
            print(f"      Direct successors of '{tool_B_name}' and their call frequencies:")
            successors_of_B = tool_graph.edges[tool_B_name]
            sorted_successors_of_B = sorted(successors_of_B.items(), key=lambda item: item[1].call_count, reverse=True)
            if not sorted_successors_of_B:
                print(f"        Tool '{tool_B_name}' has outgoing edges defined but no specific successors listed (empty target dict).") # Should not happen if edges[tool_B_name] is not empty
            for target_tool, edge_data in sorted_successors_of_B:
                print(f"        -> {target_tool}: {edge_data.call_count} calls")
        else:
            print(f"      Tool '{tool_B_name}' has no recorded successors in the graph.")

    # Analyze successors for all 2-tool sequences (tool pairs)
    print(f"\n--- Successor Analysis for All Tool Pairs (A -> B -> [Successors]) ---")
    # Data structure to store: {(tool_A, tool_B): {tool_C: frequency_of_C_after_AB}}
    tool_pair_successors_freq = defaultdict(lambda: defaultdict(int))

    if not tool_graph.paths:
        print("  No tool paths recorded, cannot analyze tool pair successors.")
    else:
        for path_obj in tool_graph.paths:
            tools_in_path = path_obj.tools
            path_frequency = path_obj.frequency
            if len(tools_in_path) >= 3:
                for i in range(len(tools_in_path) - 2):
                    tool_A = tools_in_path[i]
                    tool_B = tools_in_path[i+1]
                    tool_C = tools_in_path[i+2]
                    
                    tool_pair = (tool_A, tool_B)
                    tool_pair_successors_freq[tool_pair][tool_C] += path_frequency
        
        if not tool_pair_successors_freq:
            print("  No 3-tool sequences found in paths to analyze successors of pairs.")
        else:
            # Calculate total outgoing frequency for each (A, B) pair
            # Structure: [((A,B), total_outgoing_freq_from_AB, {C: freq, D: freq}), ...]
            pair_outgoing_analysis = []
            for pair, successors_map in tool_pair_successors_freq.items():
                total_outgoing_freq = sum(successors_map.values())
                pair_outgoing_analysis.append((pair, total_outgoing_freq, successors_map))
            
            # Sort pairs by their total outgoing frequency (i.e., how often the A->B sequence leads to *any* C)
            sorted_pairs_by_total_freq = sorted(pair_outgoing_analysis, key=lambda x: x[1], reverse=True)
            
            num_top_pairs_to_detail = 10 # How many top (A,B) pairs to detail their successors
            print(f"\n  Details for Top {min(num_top_pairs_to_detail, len(sorted_pairs_by_total_freq))} Most Frequent Tool Pairs (A -> B) and their Successors:")

            for i, (pair, total_freq, successors_map) in enumerate(sorted_pairs_by_total_freq[:num_top_pairs_to_detail]):
                print(f"\n    {i+1}. Pair: ({pair[0]} -> {pair[1]}) (This pair leads to a successor {total_freq} times in total)")
                
                # Sort successors of this specific pair by their frequency
                sorted_successors_for_this_pair = sorted(successors_map.items(), key=lambda item: item[1], reverse=True)
                
                if not sorted_successors_for_this_pair:
                    print("      No specific successors recorded for this pair (this should not happen if total_freq > 0).")
                else:
                    print("      Successors from this pair:")
                    for successor_tool, freq in sorted_successors_for_this_pair:
                        percentage = (freq / total_freq) * 100 if total_freq > 0 else 0
                        print(f"        -> {successor_tool}: {freq} times ({percentage:.1f}% of this pair's continuations)")

    # 5. 识别并打印高频工具调用边
    print(f"\n--- High-Frequency Tool Edges (Call Count >= {high_freq_tool_edge_threshold}) ---")
    high_frequency_tool_edges = []
    if not tool_graph.edges:
        print("  No tool call edges recorded in the graph.")
    else:
        for source_tool, targets in tool_graph.edges.items():
            for target_tool, edge_data in targets.items():
                if edge_data.call_count >= high_freq_tool_edge_threshold:
                    print(f"  {source_tool} -> {target_tool}: {edge_data.call_count} calls")
                    high_frequency_tool_edges.append((source_tool, target_tool))
        if not high_frequency_tool_edges:
            print(f"  No tool edges found with call count >= {high_freq_tool_edge_threshold}.")

    # --- Begin: New Section for Pie Chart Visualization ---
    print(f"\n--- Generating Successor Pie Charts ---")
    if not os.path.exists(output_pie_charts_dir):
        os.makedirs(output_pie_charts_dir)
        print(f"  Created directory: {output_pie_charts_dir}")

    # tool_pair_successors_freq is calculated in the "Successor Analysis for All Tool Pairs" section
    # Structure: tool_pair_successors_freq = defaultdict(lambda: defaultdict(int))
    # {(tool_A, tool_B): {tool_C: frequency_of_C_after_AB}}

    if not tool_pair_successors_freq and tool_pairs_for_pie_chart:
        print("  No tool pair successor data available (tool_pair_successors_freq is empty), cannot generate pie charts for pairs.")
    else:
        for pair_to_plot in tool_pairs_for_pie_chart:
            tool_A, tool_B = pair_to_plot
            if pair_to_plot in tool_pair_successors_freq:
                successors_map = tool_pair_successors_freq[pair_to_plot]
                total_freq_for_pair = sum(successors_map.values())
                if total_freq_for_pair > 0:
                    print(f"  Generating pie chart for successors of PAIR: {tool_A} -> {tool_B}")
                    plot_successor_pie_chart(pair_to_plot, successors_map, total_freq_for_pair, output_pie_charts_dir, entity_type="pair")
                else:
                    print(f"  Skipping pie chart for PAIR {tool_A} -> {tool_B}: No successors found or zero total frequency.")
            else:
                print(f"  Skipping pie chart for PAIR {tool_A} -> {tool_B}: Pair not found in analyzed tool pair successors.")

    # --- New: Generate Pie Charts for Single Tool Successors ---
    if not tool_graph.edges and tools_for_single_successor_pie_chart:
        print("  No tool edge data available (tool_graph.edges is empty), cannot generate pie charts for single tools.")
    else:
        for single_tool_to_plot in tools_for_single_successor_pie_chart:
            if single_tool_to_plot in tool_graph.edges and tool_graph.edges[single_tool_to_plot]:
                successors_of_single = tool_graph.edges[single_tool_to_plot] # This is {target_tool: ToolEdge}
                
                # Convert to the format expected by plot_successor_pie_chart: {successor_name: frequency}
                successors_map_single = {tgt: edge.call_count for tgt, edge in successors_of_single.items()}
                total_freq_for_single = sum(successors_map_single.values())
                
                if total_freq_for_single > 0:
                    print(f"  Generating pie chart for successors of SINGLE TOOL: {single_tool_to_plot}")
                    plot_successor_pie_chart(single_tool_to_plot, successors_map_single, total_freq_for_single, output_pie_charts_dir, entity_type="single")
                else:
                    print(f"  Skipping pie chart for SINGLE TOOL {single_tool_to_plot}: No successors found or zero total frequency.")
            else:
                print(f"  Skipping pie chart for SINGLE TOOL {single_tool_to_plot}: Tool not found or has no successors in tool_graph.edges.")
    # --- End: New Section for Pie Chart Visualization ---


    # --- Begin: New Section for Parameter Inertia Table ---
    # --- 核心修改：遍历列表并为每个工具生成报告 ---
    print(f"\n--- Generating Parameter Inertia Tables ---")
    if not tools_for_parameter_inertia_table:
        print("  No specific tools provided for parameter inertia analysis.")
    else:
        for i, tool_name in enumerate(tools_for_parameter_inertia_table):
            if i > 0:
                print("\n" + "="*84 + "\n") # 在多个表格之间添加一个清晰的分隔符
            
            print(f"--- Analysis for Tool: '{tool_name}' ---")
            generate_parameter_inertia_table(tool_graph, tool_name)
    # --- End: New Section for Parameter Inertia Table ---



if __name__ == "__main__":
    # --- Configuration ---
    # Please replace these paths with the actual paths to your files.
    # Ensure datastruct.py is in the same directory or accessible in PYTHONPATH.

    # Example for AlfWorld (using paths from your datastruct.py example)
    DEFAULT_TOOL_DESC_FILE = f'/root/AutoTool/AgentBoard/FastToolCalling/src/AutoTool/graph/tool_predict/tool_doc/scienceworld_tool_description.json'
    DEFAULT_TRAJECTORY_FILE = '/home/jjy/AutoTool/AgentBoard/agentboard/examples/visualisation/trajectories' # Example, may need a different file for multiple sequences or different structure

    # Use environment variables or direct paths
    tool_desc_file = os.getenv('TOOL_DESC_PATH', DEFAULT_TOOL_DESC_FILE)
    trajectory_file = os.getenv('TRAJECTORY_PATH', DEFAULT_TRAJECTORY_FILE)
    
    # Check if default files exist, otherwise prompt user (or use placeholder)

    if not os.path.exists(tool_desc_file):
        print(f"Warning: Default tool description file not found: {tool_desc_file}")
        tool_desc_file = input(f"Please enter the path to your tool description JSON file: ")
        if not os.path.exists(tool_desc_file):
            print(f"Error: Tool description file not found at '{tool_desc_file}'. Exiting.")
            exit()

    if not os.path.exists(trajectory_file):
        print(f"Warning: Default trajectory file not found: {trajectory_file}")
        trajectory_file = input(f"Please enter the path to your tool trajectory JSON file: ")
        if not os.path.exists(trajectory_file):
            print(f"Error: Trajectory file not found at '{trajectory_file}'. Exiting.")
            exit()
            
    frequency_threshold_for_tool_edges = 3 # Minimum call_count for a tool edge to be "high-frequency"
    
    # New configuration options for visualizations
    tool_pairs_for_pie_chart = [("go_to", "look_around"), ("focus_on", "wait")] # Example pairs, ensure these are likely to exist
    tools_for_parameter_inertia_table = ["use", "pick_up"] # Example tool, ensure it has input params and is used
    output_pie_charts_dir = "tool_successor_pie_charts" # Directory to save pie charts
    tools_for_single_successor_pie_chart = ["go_to", "open",] # Example single tools

    # --- End Configuration ---

    main(tool_desc_file, trajectory_file, frequency_threshold_for_tool_edges, \
         tool_pairs_for_pie_chart, tools_for_parameter_inertia_table, output_pie_charts_dir,
         tools_for_single_successor_pie_chart) 