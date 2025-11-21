# datastruct.py
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Set
import os
import json
import time
from autool.utils.embedding import get_embedding, compute_similarity
from autool.core.update.history import ExecutionHistory


class ParamEdge:
    """Parameter dependency edge between tool output and input parameters."""
    def __init__(self, source_tool: str, source_param: str, target_tool: str, target_param: str, tool_path_id: int) -> None:
        self.source_tool = source_tool
        self.source_param = source_param
        self.target_tool = target_tool
        self.target_param = target_param
        self.count = 1
        self.tool_path_id = tool_path_id

    def increment_count(self) -> None:
        self.count += 1


class ParamNode:
    """Parameter node representing a tool parameter."""
    def __init__(self, param_name: str, param_type: str = "", param_desc: str = "") -> None:
        self.name = param_name
        self.type = param_type
        self.description = param_desc
        self.is_output = False
        self.value_examples = []
        self.value_cache_size = 2

    def set_as_output(self) -> None:
        self.is_output = True
    
    def add_value_example(self, value: Any) -> None:
        if len(self.value_examples) < self.value_cache_size:
            self.value_examples.append(value)


class ParamGraph:
    """Parameter dependency graph within a tool."""
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.nodes: Dict[str, ParamNode] = {}
        self.returns: Dict[str, ParamNode] = {}
    
    def add_param_node(self, param_name: str, param_type: str = "", param_desc: str = "") -> None:
        self.nodes[param_name] = ParamNode(param_name, param_type, param_desc)
    
    def add_param_return(self, return_name: str, return_type: str = "", return_desc: str = "") -> None:
        node = ParamNode(return_name, return_type, return_desc)
        node.set_as_output()
        self.returns[return_name] = node


class ToolNode:
    """Tool node in the graph."""
    def __init__(self, tool_name: str, tool_desc: dict) -> None:
        self.name = tool_name
        self.description = tool_desc.get("tool_desc", "")
        self.args = tool_desc.get("args", [])
        self.returns = tool_desc.get("returns", {})
        
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
        
        self.param_graph = ParamGraph(tool_name)
        
        for arg_name, arg_info in self.input_params.items():
            self.param_graph.add_param_node(
                arg_name, 
                arg_info.get("type", ""), 
                arg_info.get("description", "")
            )
        
        self.param_graph.add_param_return("raw_result", "object", "Tool output result")
        
        for output_name, output_info in self.output_params.items():
            self.param_graph.add_param_return(
                output_name, 
                output_info.get("type", ""), 
                output_info.get("description", "")
            )
    
    def format_tool_description(self) -> str:
        """Format tool description consistent with toolkit format."""
        tool_desc = f"{self.name}: {self.description}\n"
        
        for arg in self.args:
            arg_name = arg.get("arg_name", "")
            arg_type = arg.get("arg_type", "")
            arg_desc = arg.get("arg_desc", "")
            tool_desc += f"{arg_name}({arg_type}): {arg_desc}\n"
        
        tool_desc += "\n"
        return tool_desc


class ToolEdge:
    """Edge connecting two tool nodes."""
    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target
        self.call_count = 0
        self.param_mappings = {}
    
    def increment_call(self) -> None:
        self.call_count += 1
    
    def add_param_mapping(self, target_param: str, source_field: str) -> None:
        self.param_mappings[target_param] = source_field


class ToolPath:
    """Complete tool call path."""
    def __init__(self, tools: List[str]) -> None:
        self.tools = list(tools) if not isinstance(tools, list) else tools
        self.frequency = 1
        self.confidence = 0.0
    
    def __eq__(self, other):
        if isinstance(other, ToolPath):
            return self.tools == other.tools
        elif isinstance(other, list):
            return self.tools == other
        return False
    
    def __hash__(self):
        return hash(tuple(self.tools))
    
    def update_frequency(self, weight) -> None:
        self.frequency += weight
    
    def match_prefix(self, prefix: List[str]) -> bool:
        if len(prefix) > len(self.tools):
            return False
        return self.tools[:len(prefix)] == prefix
    
    def get_subsequent_path(self, prefix_length: int) -> List[str]:
        if prefix_length >= len(self.tools):
            return []
        return self.tools[prefix_length:]


class ToolGraph:
    """Tool call graph with tools as nodes and call chains as edges."""
    def __init__(self) -> None:
        # 基础结构
        self.nodes: Dict[str, ToolNode] = {}  # 工具节点字典 {tool_name: ToolNode}
        self.edges: Dict[str, Dict[str, ToolEdge]] = {}  # 工具调用边字典 {source: {target: ToolEdge}}
        
        # 路径管理
        self.paths: List[ToolPath] = []  # 工具调用路径集合，每个唯一路径只存储一次
        self.tooll2chainID: Dict[str, set[int]] = defaultdict(set)  # 索引：工具名到路径ID的映射 {tool_name: {path_id1, path_id2, ...}}
        self.chain2ID: Dict[Tuple[str, ...], int] = {}  # 查找表：路径序列到路径ID的映射，用于快速查找路径是否已存在
        
        # 新增：参数依赖边存储
        # 结构: {target_tool: {target_param: {(source_tool, source_param_name): ParamEdge}}}
        self.param_edges: Dict[str, Dict[str, Dict[Tuple[str, str], ParamEdge]]] = defaultdict(lambda: defaultdict(dict))
        
        self.task_description = ""
        self.previous_progress = ""
        self.debug = False

    def add_node(self, tool_name: str, tool_desc: dict) -> None:
        self.nodes[tool_name] = ToolNode(tool_name, tool_desc)
        if tool_name not in self.edges:
            self.edges[tool_name] = {}
            self.tooll2chainID[tool_name] = set()
    
    def add_edge(self, source: str, target: str) -> None:
        if source not in self.nodes:
            raise ValueError(f"Source tool {source} not found in graph")
        if target not in self.nodes:
            raise ValueError(f"Target tool {target} not found in graph")
        
        if target not in self.edges[source]:
            self.edges[source][target] = ToolEdge(source, target)
    
    def record_tool_sequence(self, tool_sequence: List[str], weight=1) -> None:
        """Update tool call chains and edges from complete tool sequence."""
        if not tool_sequence or len(tool_sequence) < 3:
            print("Warning: Tool sequence too short to record")
            return

        for i in range(len(tool_sequence) - 1):
            source = tool_sequence[i]
            target = tool_sequence[i + 1]
            
            if source not in self.edges or target not in self.edges.get(source, {}):
                if source not in self.nodes:
                    self.add_node(source, {})
                if target not in self.nodes:
                    self.add_node(target, {})
                self.add_edge(source, target)
            
            self.edges[source][target].increment_call()
        print(f"[ToolGraph] Recorded edges for sequence: {tool_sequence}")
        self._update_path(tool_sequence, weight=weight)

    def _update_path(self, tool_sequence: List[str], weight=1) -> None:
        tool_sequence_tuple = tuple(tool_sequence)
        
        if self.debug:
            print(f'[ToolGraph] Updating path: {tool_sequence}, weight: {weight}')
            
        if tool_sequence_tuple in self.chain2ID:
            path_id = self.chain2ID[tool_sequence_tuple]
            self.paths[path_id].update_frequency(weight)
            return
        
        path_id = len(self.paths)
        new_path = ToolPath(tool_sequence)
        self.paths.append(new_path)
        self.chain2ID[tool_sequence_tuple] = path_id
        
        for tool in tool_sequence:
            self.tooll2chainID[tool].add(path_id)

    def to_json(self) -> str:
        graph_data = {
            "nodes": {},
            "edges": [],
            "paths": []
        }
        
        for tool_name, node in self.nodes.items():
            graph_data["nodes"][tool_name] = {
                "name": node.name,
                "description": node.description,
                "input_params": node.input_params
            }
        
        for source, targets in self.edges.items():
            for target, edge in targets.items():
                graph_data["edges"].append({
                    "source": source,
                    "target": target,
                    "call_count": edge.call_count,
                    "param_mappings": edge.param_mappings
                })
        
        for path in self.paths:
            graph_data["paths"].append({
                "tools": path.tools,
                "frequency": path.frequency
            })
        
        return json.dumps(graph_data, indent=2, ensure_ascii=False)
    
    # def get_potential_path_indices(self, matched_tool_sequence):
        # print(f"[DEBUG] Getting potential path indices for sequence: {matched_tool_sequence}")
        # if matched_tool_sequence[0] in self.tooll2chainID:
        #     potential_path_indices = self.tooll2chainID[matched_tool_sequence[0]].copy()
        # else:
        #     return set()
        
        # for i in range(1, len(matched_tool_sequence)):
        #     if matched_tool_sequence[i] not in self.tooll2chainID:
        #         return set()
        #     potential_path_indices &= self.tooll2chainID[matched_tool_sequence[i]]
        
        # return potential_path_indices
    def get_potential_path_indices(self, matched_tool_sequence):
        print(f"[DEBUG] Getting potential path indices for sequence: {matched_tool_sequence}")
        
        if isinstance(matched_tool_sequence, str):
            matched_tool_sequence = [matched_tool_sequence]
        
        if not matched_tool_sequence:
            return set()
        
        # 【添加】检查第一个元素
        first_tool = matched_tool_sequence[0]
        print(f"[DEBUG] First tool: {first_tool}, type: {type(first_tool)}")
        print(f"[DEBUG] path_index keys (first 5): {list(self.tooll2chainID.keys())[:5]}")
        
        if first_tool in self.tooll2chainID:
            potential_path_indices = self.tooll2chainID[first_tool].copy()
            print(f"[DEBUG] Found {len(potential_path_indices)} potential paths for '{first_tool}'")
        else:
            print(f"[DEBUG] No paths found for '{first_tool}'")
            return set()
        
        for i in range(1, len(matched_tool_sequence)):
            tool = matched_tool_sequence[i]
            print(f"[DEBUG] Checking tool at index {i}: {tool}")
            
            if tool not in self.tooll2chainID:
                print(f"[DEBUG] Tool '{tool}' not in path_index")
                return set()
            
            potential_path_indices &= self.tooll2chainID[tool]
            print(f"[DEBUG] After intersecting '{tool}': {len(potential_path_indices)} paths remain")
        
        print(f"[DEBUG] Final potential_path_indices: {potential_path_indices}")
        return potential_path_indices

    def get_param_sources(self, matched_tool_sequence, target_tool, miss_param_name):
        """
        Find inertial sources for missing parameters based on param_edges.
        
        Returns:
            List of tuples (source_tool, source_param, count) sorted by count descending
        """
        results = []
        
        if target_tool not in self.param_edges or miss_param_name not in self.param_edges[target_tool]:
            return results
        
        param_source_edges = self.param_edges[target_tool][miss_param_name]
        id_set = self.get_potential_path_indices(matched_tool_sequence)

        for (source_tool, source_param, path_id), param_edge in param_source_edges.items():
            if path_id in id_set:
                results.append((source_tool, source_param, param_edge.count))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def attach_param_edge(self, history, tool_name, inputs, current_sequence):
        """Attach parameter edges to graph's param_edges attribute."""
        if len(current_sequence) < 2:
            return

        if tool_name not in self.nodes:
            print(f"Warning: Target tool '{tool_name}' not in graph")
            return
            
        if not inputs:
            return
        
        tool_node = self.nodes[tool_name]
        param_graph = tool_node.param_graph
        
        for target_param_name, param_value in inputs.items():
            if param_value is None:
                continue
                
            if target_param_name in param_graph.nodes:
                if self.debug:
                    print(f'[ToolGraph] Adding value example for {tool_name}.{target_param_name}: {param_value}')
                param_graph.nodes[target_param_name].add_value_example(param_value)
            
            for record in reversed(history.history):
                source_tool_name = record.get("tool_name")
                outputs = record.get("outputs", {})
                inputs_rec = record.get("inputs", {})
                
                if source_tool_name == tool_name or source_tool_name == "unknown" or (not outputs and not inputs_rec):
                    continue
                
                if source_tool_name not in self.nodes:
                    pass
                    
                matched_source_param = None

                def is_match(value, param_value):
                    if value == param_value:
                        return True
                    if isinstance(value, list) and param_value in value:
                        return True
                    if isinstance(value, dict):
                        if param_value in value or param_value in value.values():
                            return True
                    if isinstance(value, set) and param_value in value:
                        return True
                    return False

                if outputs:
                    for source_param_name, output_value in outputs.items():
                        if is_match(output_value, param_value):
                            for input_param_name, input_value in inputs_rec.items():
                                if is_match(input_value, param_value):
                                    matched_source_param = (source_param_name, output_value)
                                    break
                            if matched_source_param:
                                break
                else:
                    for input_param_name, input_value in inputs_rec.items():
                        if is_match(input_value, param_value):
                            matched_source_param = (input_param_name, input_value)
                            break
        
                tool_sequence_tuple = tuple(current_sequence)
                potential_path_indices = self.get_potential_path_indices(tool_sequence_tuple)
                list_id = list(potential_path_indices)
                
                for path_id in list_id:
                    if matched_source_param is not None:
                        param_edge_key = (source_tool_name, matched_source_param[0])
                        target_param_dict = self.param_edges[tool_name][target_param_name]
                        
                        if param_edge_key in target_param_dict:
                            if self.debug:
                                print(f"  Updating ParamEdge: ({source_tool_name}, {matched_source_param}) -> ({tool_name}, {target_param_name}), count={target_param_dict[param_edge_key].count}")
                            target_param_dict[param_edge_key].increment_count()
                        else:
                            if self.debug:
                                print(f"  Creating ParamEdge: ({source_tool_name}, {matched_source_param}) -> ({tool_name}, {target_param_name}), count=1")
                            new_param_edge = ParamEdge(source_tool_name, matched_source_param, tool_name, target_param_name, path_id)
                            target_param_dict[param_edge_key] = new_param_edge
                    break
                    
    def update_graph(self, current_sequence: Dict) -> None:
        """Update graph structure from current_sequence, supporting sequence splitting at inertial steps."""
        if not current_sequence:
            print("Warning: Empty current_sequence provided")
            return
            
        history = ExecutionHistory(max_len=40)
        steps = current_sequence.get("steps", [])
        
        if not steps:
            print("Warning: No steps found in current_sequence")
            return
            
        inertial_steps = current_sequence.get("inertial_step", [])
        inertial_indices = set()
        
        for inertial_step in inertial_steps:
            inertial_indices.add(inertial_step[0])
        
        tool_sequences = []
        current_sequence_tools = []
        
        for i, step in enumerate(steps):
            if step.get("type") == "initial":
                continue
                
            if i in inertial_indices:
                if current_sequence_tools:
                    tool_sequences.append(current_sequence_tools)
                current_sequence_tools = []
                continue

            if step.get("type") == "act_ob_pair":
                action_data = step.get("action", {})
                observation_data = step.get("observation", {})
                parsed_observation = observation_data.get("parsed_content", None)
                parsed_action = action_data.get("parsed_content", None)
                
                tool_name = None
                if parsed_action and isinstance(parsed_action, dict) and parsed_action.get("tool_name") not in ["unknown", None]:
                    tool_name = parsed_action.get("tool_name")
                    current_sequence_tools.append(tool_name)
                    self.attach_param_edge(history, tool_name, parsed_action.get("inputs", {}), current_sequence_tools)
                    
                    record = {
                        "tool_name": tool_name,
                        "inputs": parsed_action.get("inputs", {}),
                        "outputs": parsed_observation,
                    }
                    history.add_record(record)
        
        if current_sequence_tools:
            tool_sequences.append(current_sequence_tools)
        
        total_tools = 0
        if self.debug:
            print(f"Identified {len(tool_sequences)} sub-sequences after splitting")

        for idx, sequence in enumerate(tool_sequences):
            if len(sequence) > 1:
                if self.debug:
                    print(f"  - Updating graph with sub-sequence {idx+1}: {sequence}")
                try:
                    self.record_tool_sequence(sequence)
                    total_tools += len(sequence)
                except Exception as e:
                    print(f"Error updating sub-sequence {idx+1}: {e}")
            else:
                if self.debug:
                    print(f"  - Skipping sub-sequence {idx+1} with length {len(sequence)}")
        
        if self.debug:
            print(f"Graph update complete: {len(self.paths)} paths, {total_tools} tools")
            print(f"Path lookup table size: {len(self.chain2ID)}")

    def load_from_json(self, tool_path: str, chain_path: str) -> None:
        self.load_tool_description_from_json(tool_path)
        self.load_tool_chain_from_json(chain_path)
        print(f"Tool Graph loaded from {tool_path} and {chain_path}")
    
    def load_tool_chain_from_json(self, chain_path: str) -> None:
        if not os.path.isabs(chain_path):
            chain_path = os.path.join(os.path.dirname(__file__), chain_path)
            
        try:
            if os.path.getsize(chain_path) == 0:
                print("Starting from empty trajectory")
                return
            else:
                with open(chain_path, "r", encoding="utf-8") as f:
                    tool_chains = json.load(f)
                    tool_chains = tool_chains.get("sequences", {})
                    print(f"Loading {len(tool_chains)} tool call sequences")
            
            for trajectory in tool_chains:
                tool_sequence = []
                steps = trajectory.get("steps", [])
                for step in steps:
                    if step.get("type") == "act_ob_pair":
                        action = step.get("action", {})
                        parsed_content = action.get("parsed_content", {})
                        tool_name = parsed_content.get("tool_name")
                        if tool_name and tool_name != "unknown":
                            tool_sequence.append(tool_name)

                if tool_sequence:
                    self.record_tool_sequence(tool_sequence)

            print(f"Loaded {len(self.paths)} paths")
            print(f"Path lookup table size: {len(self.chain2ID)}")
            
        except Exception as e:
            print(f"Failed to load chains from JSON: {e}")
            import traceback
            traceback.print_exc()

    def load_tool_description_from_json(self, tool_path: str) -> None:
        if not os.path.isabs(tool_path):
            tool_path = os.path.join(os.path.dirname(__file__), tool_path)
        
        try:
            with open(tool_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
                print(f"Loading {len(graph_data)} tools")

            for tool_name, node_data in graph_data.items():
                tool_desc = {
                    "tool": tool_name,
                    "tool_desc": node_data.get("tool_desc", ""),
                    "args": [],
                    "returns": []
                }
                
                args = node_data.get("args", [])
                returns = node_data.get("returns", [])
                
                if args:
                    for param in args:
                        tool_desc["args"].append(param)
                        
                if returns:
                    for param in returns:
                        tool_desc["returns"].append(param)

                self.add_node(tool_name, tool_desc)

            print(f"Loaded {len(self.nodes)} tools")
            
        except Exception as e:
            print(f"Failed to load nodes from JSON: {e}")
            import traceback
            traceback.print_exc()

    def _get_path_text_representation(self, path: List[str], end_pos: int, ROI_len: int = 1) -> str:
        path_texts = []
        tools_info = []
        
        for tool_name in path[end_pos:end_pos+ROI_len]:
            if tool_name in self.nodes:
                tools_info.append({
                    "name": tool_name,
                    "desc": self.nodes[tool_name].description
                })
        
        if len(tools_info) > 0:
            for i, tool in enumerate(tools_info):
                prefix = f"Step{i+1}: "
                tool_desc = tool['desc'] if tool['desc'] else "no description"
                tool_text = f"{prefix}use {tool['name']}({tool_desc})"
                path_texts.append(tool_text)
            
            if len(tools_info) > 1:
                path_texts.append(f"Call chain from {tools_info[0]['name']} to {tools_info[-1]['name']}")
        
        return "\n".join(path_texts)

    def _get_confidence(self, x) -> float:
        return 1 - pow(1.1, -x)

    def predict_next_tool_with_chain_similarity(self, current_sequence: List[str], intuition: str = "", threshold=0.4, alpha=0.5):
        """Predict next tool by computing semantic similarity of tool chains."""
        print("[DEBUG]: Starting graph prediction...")
        overhead = defaultdict(float)

        if not current_sequence:
            return False, overhead, ""
        print(f"[DEBUG] current_sequence type: {type(current_sequence)}, value: {current_sequence}")
        s_time = time.time()
        potential_path_indices = self.get_potential_path_indices(current_sequence)
        candidate_tool_dic = defaultdict(float)
        total_frequency = 0
        
        if self.debug:
            print(f'[DEBUG] potential_path_indices: {potential_path_indices}')
            
        if not potential_path_indices:
            print("Warning: No potential paths found for the current sequence")
            print("No matching paths found")
            return False, overhead, ""
        # 【添加】在循环前检查
        print(f"[DEBUG] Processing {len(potential_path_indices)} paths: {potential_path_indices}")

        for path_idx in potential_path_indices:
            if path_idx >= len(self.paths):
                print(f"Warning: Path index {path_idx} out of range")
                continue
            
            path = self.paths[path_idx]
            # 【添加】检查 path.tools 类型
            print(" [DEBUG] Checking path.tools...")
            print(f"    path_idx: {path_idx}, type: {type(path.tools)}, value: {path.tools}")
            print(f"   path.tools: {path.tools}")
            is_subseq, start_pos = is_subsequence(current_sequence, path.tools)
            print(f"[DEBUG] Path {path_idx}: is_subseq={is_subseq}, start_pos={start_pos}, path.tools={path.tools}")
            if is_subseq and start_pos + len(current_sequence) < len(path.tools):
                print(f"[DEBUG] Path {path_idx} is a subsequence match at position {start_pos}")
                end_pos = start_pos + len(current_sequence)
                next_tool = path.tools[end_pos]
                candidate_tool_dic[next_tool] += path.frequency
                total_frequency += path.frequency
                print(f"[DEBUG] Added candidate tool: {next_tool}, frequency={path.frequency}")
                if self.debug:
                    print(f'[DEBUG] next tool {next_tool}, frequency {path.frequency}')
        
        next_tool_scores = {}
        
        if self.debug:
            print(f'[DEBUG] total_frequency: {total_frequency}')
            
        for tool_name, frequency in candidate_tool_dic.items():
            if tool_name not in next_tool_scores:
                next_tool_scores[tool_name] = {
                    "frequency_score": 0,
                    "semantic_score": 0,
                    "combined_score": 0
                }
            next_tool_scores[tool_name]["frequency_score"] = (frequency / total_frequency) * self._get_confidence(total_frequency)
            
        search_time = time.time() - s_time
        s_time = time.time()
        intuition_embedding_time = 0.0
        path_embedding_time = 0.0
        semantic_score_time = 0.0
        
        if intuition is None:
            intuition = self.task_description
            
        if intuition and intuition.strip():
            intuition_embedding = get_embedding(intuition)
            intuition_embedding_time = time.time() - s_time
            
            s1_time = time.time()
            
            for tool_name, _ in candidate_tool_dic.items():
                tool_text = f"{tool_name}: {self.nodes[tool_name].description}"

                if self.debug:
                    print(f'[DEBUG] path_text: {tool_text}')
                    
                path_embedding = get_embedding(tool_text)
                path_embedding_time += time.time() - s1_time
                
                s2_time = time.time()
                semantic_score = compute_similarity(path_embedding, intuition_embedding)
                semantic_score_time += time.time() - s2_time
                next_tool_scores[tool_name]["semantic_score"] = semantic_score
        
        s_time = time.time()
        for tool_name, scores in next_tool_scores.items():
            combined_score = alpha * scores["frequency_score"] + (1 - alpha) * scores["semantic_score"]
            scores["combined_score"] = combined_score
            
        sorted_path_scores = sorted(next_tool_scores.items(), key=lambda x: x[1]['combined_score'], reverse=True)

        overhead["search_time"] = search_time
        overhead["intuition_embedding_time"] = intuition_embedding_time
        overhead["path_embedding_time"] = path_embedding_time
        overhead["similarity_cost_time"] = semantic_score_time
        overhead["SimSCE_time"] = semantic_score_time + path_embedding_time + intuition_embedding_time
        
        if not sorted_path_scores:
            print("No matching paths found")
            return False, overhead, ""
            
        best_tool_name = sorted_path_scores[0][0]
        best_score = sorted_path_scores[0][1]['combined_score']
        
        if not best_tool_name:
            print("Cannot determine best path")
            return False, overhead, ""
            
        if best_score > threshold:
            print(f"Inertial call detected, score: {best_score:.4f}, threshold: {threshold}")
            return True, overhead, best_tool_name
        
        print(f"Score insufficient for inertial call, score: {best_score:.4f}, threshold: {threshold}, tool: {best_tool_name}")
        return False, overhead, ""
    
    # def predict_next_tool_with_chain_similarity(self, current_sequence: List[str], intuition: str = "", threshold=0.4, alpha=0.5):
    #     """Predict next tool by computing semantic similarity of tool chains."""
    #     overhead = defaultdict(float)

    #     if not current_sequence:
    #         return False, overhead, ""
        
    #     s_time = time.time()
    #     potential_path_indices = self.get_potential_path_indices(current_sequence)
    #     candidate_tool_dic = defaultdict(float)
    #     total_frequency = 0
        
    #     if not potential_path_indices:
    #         return False, overhead, ""

    #     # 【添加】计数匹配的路径
    #     matched_paths = 0
        
    #     for path_idx in potential_path_indices:
    #         if path_idx >= len(self.paths):
    #             continue
                
    #         path = self.paths[path_idx]
            
    #         try:
    #             is_subseq, start_pos = is_subsequence(current_sequence, path.tools)
                
    #             # 【添加】输出关键信息
    #             if is_subseq:
    #                 print(f"[DEBUG] Path {path_idx}: is_subseq=True, start_pos={start_pos}, path.tools={path.tools}")
                    
    #             if is_subseq and start_pos + len(current_sequence) < len(path.tools):
    #                 end_pos = start_pos + len(current_sequence)
    #                 next_tool = path.tools[end_pos]
    #                 candidate_tool_dic[next_tool] += path.frequency
    #                 total_frequency += path.frequency
    #                 matched_paths += 1
    #                 print(f"[DEBUG] Added candidate: {next_tool}, frequency={path.frequency}")
    #         except (TypeError, IndexError) as e:
    #             print(f"[DEBUG] Exception for path {path_idx}: {e}")
    #             continue
        
    #     # 【添加】输出最终结果
    #     print(f"[DEBUG] Matched {matched_paths} paths, candidates: {dict(candidate_tool_dic)}")
        
    #     if not candidate_tool_dic:
    #         return False, overhead, ""
        
        # ... 后续代码不变


    def evaluate_candidate_tools(self, 
                                 current_sequence: List[str], 
                                 candidate_tools: List[str],
                                 intuition: str = "", 
                                 alpha: float = 0.5) -> Dict[str, Dict[str, float]]:
        """
        Evaluate candidate tools using hierarchical fallback strategy.
        Prioritize longer history, fallback to shorter only when no matches.
        """
        final_scores = {tool: {"frequency_score": 0.0, "semantic_score": 0.0, "combined_score": 0.0} 
                       for tool in candidate_tools}

        if not candidate_tools:
            return final_scores

        history_len_2 = current_sequence[-2:] if len(current_sequence) >= 2 else []
        history_len_1 = current_sequence[-1:] if len(current_sequence) >= 1 else []

        def get_frequencies(history):
            if not history:
                return defaultdict(float), 0
            
            potential_path_indices = self.get_potential_path_indices(history)
            if not potential_path_indices:
                return defaultdict(float), 0

            frequencies = defaultdict(float)
            total_freq = 0
            
            for path_idx in potential_path_indices:
                path = self.paths[path_idx]
                is_sub, start_pos = is_subsequence(history, path.tools)
                if is_sub and start_pos + len(history) < len(path.tools):
                    next_tool = path.tools[start_pos + len(history)]
                    if next_tool in candidate_tools:
                        frequencies[next_tool] += path.frequency
                        total_freq += path.frequency
            return frequencies, total_freq

        candidate_frequencies_len_2, total_frequency_len_2 = get_frequencies(history_len_2)
        candidate_frequencies_len_1, total_frequency_len_1 = get_frequencies(history_len_1)

        final_frequencies = {}
        final_total_frequency = 0

        if total_frequency_len_2 > 0:
            if self.debug:
                print("[Fallback] Using primary history of length 2")
            final_frequencies = candidate_frequencies_len_2
            final_total_frequency = total_frequency_len_2
        elif total_frequency_len_1 > 0:
            if self.debug:
                print("[Fallback] Using history of length 1")
            final_frequencies = candidate_frequencies_len_1
            final_total_frequency = total_frequency_len_1
        else:
            if self.debug:
                print("[Fallback] No matching history found")
            return final_scores

        confidence_factor = self._get_confidence(final_total_frequency)
        for tool_name in candidate_tools:
            frequency = final_frequencies.get(tool_name, 0.0)
            if final_total_frequency > 0:
                final_scores[tool_name]["frequency_score"] = (frequency / final_total_frequency) * confidence_factor

        if intuition and intuition.strip():
            try:
                intuition_embedding = get_embedding(intuition)
                for tool_name in candidate_tools:
                    if tool_name in self.nodes:
                        tool_text = f"{tool_name}: {self.nodes[tool_name].description}"
                        path_embedding = get_embedding(tool_text)
                        semantic_score = compute_similarity(path_embedding, intuition_embedding)
                        final_scores[tool_name]["semantic_score"] = semantic_score
            except Exception as e:
                print(f"Warning: Error during semantic score calculation: {e}")

        for tool_name in candidate_tools:
            scores = final_scores[tool_name]
            combined_score = (alpha * scores["frequency_score"]) + ((1 - alpha) * scores["semantic_score"])
            final_scores[tool_name]["combined_score"] = combined_score
            
        return final_scores


# def is_subsequence(subseq: List[str], seq: List[str]) -> Tuple[bool, int]:
#     if len(subseq) > len(seq):
#         return False, -1
    
#     for i in range(len(seq) - len(subseq) + 1):
#         if seq[i:i+len(subseq)] == subseq:
#             return True, i
    
#     return False, -1

def is_subsequence(subseq: List[str], seq: List[str]) -> Tuple[bool, int]:
    """
    检查 subseq 是否是 seq 的子序列
    返回 (是否匹配, 起始位置)
    """
    # 【添加】类型检查
    if not isinstance(subseq, list):
        subseq = list(subseq) if hasattr(subseq, '__iter__') else [subseq]
    if not isinstance(seq, list):
        seq = list(seq) if hasattr(seq, '__iter__') else [seq]
    
    if not subseq or not seq:
        return False, -1
    
    subseq_len = len(subseq)
    seq_len = len(seq)
    
    if subseq_len > seq_len:
        return False, -1
    
    # 滑动窗口匹配
    for i in range(seq_len - subseq_len + 1):
        # 【修复】确保都是列表后再切片
        if seq[i:i+subseq_len] == subseq:
            return True, i
    
    return False, -1