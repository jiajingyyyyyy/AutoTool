from typing import Dict, List, Any, Tuple, Optional, Set
import json
import os
import re
from collections import defaultdict
class ParameterDependencyGraph:
    """参数依赖图，管理工具之间的参数依赖关系"""
    
    def __init__(self, data_path: Optional[str] = None) -> None:
        """
        初始化参数依赖图
        
        Args:
            data_path: 依赖图数据路径，如果提供则从文件加载
        """
        # 参数依赖关系: {(source_tool, source_param, 'output'): {(target_tool, target_param, 'input'): count}}
        self.dependencies: Dict[Tuple[str, str, str], Dict[Tuple[str, str, str], int]] = defaultdict(lambda: defaultdict(int))
        
        # 如果提供了数据路径，尝试加载
        if data_path:
            self.load_from_file(data_path)
            
        self.debug = False
    
    def set_debug(self, debug: bool) -> None:
        """设置调试模式"""
        self.debug = debug
    
    def add_dependency(self, 
                      source_tool: str, source_param: str, 
                      target_tool: str, target_param: str, 
                      count: int = 1) -> None:
        """
        添加参数依赖关系
        
        Args:
            source_tool: 源工具名称
            source_param: 源参数名称
            target_tool: 目标工具名称
            target_param: 目标参数名称
            count: 依赖关系计数，默认为1
        """
        source_key = (source_tool, source_param, 'output')
        target_key = (target_tool, target_param, 'input')
        
        self.dependencies[source_key][target_key] += count
    
    def get_potential_sources(self, target_tool: str, target_param: str) -> List[Tuple[str, str, int]]:
        """
        获取目标参数的潜在源参数
        
        Args:
            target_tool: 目标工具名称
            target_param: 目标参数名称
            
        Returns:
            潜在源参数列表，每项包含 (工具名, 参数名, 计数)
        """
        target_key = (target_tool, target_param, 'input')
        potential_sources = []
        
        for source_key, targets in self.dependencies.items():
            if target_key in targets:
                source_tool, source_param, _ = source_key
                count = targets[target_key]
                potential_sources.append((source_tool, source_param, count))
        
        # 按计数降序排序
        potential_sources.sort(key=lambda x: x[2], reverse=True)
        return potential_sources
    
    def improved_value_match_func(self, v1: Any, v2: Any) -> bool:
        """
        改进的值匹配函数，处理字符串相等、列表包含等情况。
        Args:
            v1: 第一个值 (通常是输出值)
            v2: 第二个值 (通常是输入值)
        Returns:
            如果两个值匹配，返回 True，否则返回 False
        """
        if v1 is None or v2 is None:
            return False

        # 1. 严格相等（处理基础类型如字符串、数字、布尔值）
        if v1 == v2:
            return True

        # 尝试将 v1 或 v2 解析为列表/字典，以防它们是字符串形式的表示
        # （这取决于你的执行历史中 outputs/inputs 是原始类型还是字符串化的）
        try:
            v1_parsed = json.loads(v1) if isinstance(v1, str) and (v1.strip().startswith('[') or v1.strip().startswith('{')) else v1
        except (json.JSONDecodeError, TypeError):
            v1_parsed = v1

        try:
            v2_parsed = json.loads(v2) if isinstance(v2, str) and (v2.strip().startswith('[') or v2.strip().startswith('{')) else v2
        except (json.JSONDecodeError, TypeError):
            v2_parsed = v2

        # 1.1 再次尝试严格相等（可能解析后匹配了）
        if v1_parsed == v2_parsed:
            return True

        # 2. 列表包含关系：如果 v1 是一个列表，检查 v2 是否在 v1 中
        # 同时也考虑 v1 是字符串化的列表，v1_parsed 是列表的情况
        if isinstance(v1_parsed, (list, tuple, set)) and not isinstance(v1_parsed, str): # 确保不是字符串伪装的列表
            # 检查 v2 或其解析后的形式是否在 v1_parsed 中
            if v2 in v1_parsed:
                return True
            if v2_parsed is not v2 and v2_parsed in v1_parsed: # 如果v2也被解析过，也检查解析后的形式
                return True

        # 3. (可选) 字符串包含关系：如果 v1 是字符串，检查 v2 是否在 v1 中（仅在特定tool-query场景下可能需要）
        # 例如，一个工具返回一段描述性文字，后续工具的输入是其中的一个关键词
        # if isinstance(v1_parsed, str) and isinstance(v2_parsed, str) and v2_parsed in v1_parsed:
        #     return True

        # 4. (可选) 其他自定义匹配逻辑...

        # 如果以上都不匹配
        return False

    def build_from_execution_history(self, execution_history: List[Dict[str, Any]], 
                                    value_match_func=None, ROI=5, with_input=True) -> None:
        """
        从执行历史记录构建参数依赖关系
        
        Args:
            execution_history: 执行历史记录列表
            value_match_func: 值匹配函数，用于判断两个值是否匹配
        """
        print(f"[DEBUG ] build_from_execution_history: {len(execution_history)} records")
        if not execution_history:
            return
            
        if not value_match_func:
            # 默认的值匹配函数，简单的字符串比较
            value_match_func = lambda v1, v2: str(v1) == str(v2)
        
        # 遍历执行历史
        for i, record in enumerate(execution_history[:-1]):
            # 当前工具信息
            curr_tool = record.get('tool_name')
            curr_outputs = record.get('outputs', {})
            curr_inputs = record.get('inputs', {})
            if not curr_tool:
                continue
            
            

            # 合并输入和输出未 curr_node变量
            if with_input:
                curr_node = {**curr_outputs, **curr_inputs}
            else:
                curr_node = curr_outputs
                
            # print(f"[DEBUG] record {i} tool : {curr_tool}: curr_node: {curr_node}")
            # 检查后续工具
            for j in range(i+1, min(i + ROI, len(execution_history))):
                next_record = execution_history[j]
                next_tool = next_record.get('tool_name')
                next_inputs = next_record.get('inputs', {})
                # print(f"[DEBUG ] record {j}: next_tool: {next_tool} next_inputs: {next_inputs}")
                if not next_tool or not next_inputs:
                    continue
                    
                # 检查参数依赖关系
                for out_name, out_value in curr_node.items():
                    if out_value is None:
                        continue
                        
                    for in_name, in_value in next_inputs.items():
                        if in_value is None:
                            continue
                        # 检查值是否匹配
                        if value_match_func(out_value, in_value):
                            print(f"[DEBUG ] build from execution history: {curr_tool}.{out_name} -> {next_tool}.{in_name}")
                            self.add_dependency(curr_tool, out_name, next_tool, in_name)
                            if self.debug:
                                print(f"添加依赖: ({curr_tool}.{out_name}) -> ({next_tool}.{in_name})")
    
    def save_to_file(self, file_path: str):
        """将依赖图保存为JSON文件"""
        # 转换为可序列化格式
        serializable_deps = {}
        for src_tup, targets in self.dependencies.items():
            # 将元组键转为字符串表示
            src_key = str(src_tup)
            serializable_deps[src_key] = {}
            
            for tgt_tup, weight in targets.items():
                tgt_key = str(tgt_tup)
                serializable_deps[src_key][tgt_key] = weight
        
        # 确保目录存在
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            print(f"创建目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_deps, f, indent=2, ensure_ascii=False)
                
            if self.debug:
                print(f"依赖图已保存到 {file_path}")
        except Exception as e:
            print(f"保存依赖图失败: {e}")
    
    def load_from_file(self, file_path: str) -> None:
        """
        从文件加载依赖图
        
        Args:
            file_path: 文件路径
        """
        if not os.path.exists(file_path):
            if self.debug:
                print(f"依赖图文件不存在: {file_path}")
                print("不执行加载操作")
            return
            
        try:
            if os.path.getsize(file_path) == 0:
                if self.debug:
                    print(f"依赖图文件为空: {file_path}")
                return
            with open(file_path, 'r', encoding='utf-8') as f:
                serialized_deps = json.load(f)
                
            # 解析并恢复依赖关系
            self.dependencies.clear()
            for source_str, targets in serialized_deps.items():
                try:
                    # 处理字符串格式的元组
                    source_tuple = eval(source_str)
                    
                    for target_str, count in targets.items():
                        target_tuple = eval(target_str)
                        self.dependencies[source_tuple][target_tuple] = count
                except:
                    if self.debug:
                        print(f"解析依赖项失败: {source_str}")
                    continue
                    
            if self.debug:
                print(f"从 {file_path} 加载了 {len(self.dependencies)} 个依赖项")
        except Exception as e:
            print(f"加载依赖图失败: {e}")
    
    def merge(self, other_graph) -> None:
        """
        合并另一个依赖图
        
        Args:
            other_graph: 另一个ParameterDependencyGraph实例
        """
        for source, targets in other_graph.dependencies.items():
            for target, count in targets.items():
                self.dependencies[source][target] += count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取依赖图统计信息
        
        Returns:
            统计信息字典
        """
        source_tools = set()
        target_tools = set()
        total_deps = 0
        total_counts = 0
        
        for source, targets in self.dependencies.items():
            source_tool = source[0]
            source_tools.add(source_tool)
            
            for target, count in targets.items():
                target_tool = target[0]
                target_tools.add(target_tool)
                total_deps += 1
                total_counts += count
        
        return {
            "source_tools_count": len(source_tools),
            "target_tools_count": len(target_tools),
            "total_dependencies": total_deps,
            "total_counts": total_counts,
            "unique_tools": len(source_tools.union(target_tools))
        }
    
    def __str__(self) -> str:
        """
        字符串表示
        
        Returns:
            依赖图的字符串表示
        """
        stats = self.get_stats()
        return f"ParameterDependencyGraph(工具: {stats['unique_tools']}, 依赖项: {stats['total_dependencies']})"
    
    def visualize(self, top_n: int = 10) -> None:
        """
        可视化依赖图中最常见的依赖关系(简单命令行版本)
        
        Args:
            top_n: 显示前N个最频繁的依赖关系
        """
        all_deps = []
        
        for source, targets in self.dependencies.items():
            source_tool, source_param, _ = source
            
            for target, count in targets.items():
                target_tool, target_param, _ = target
                all_deps.append((f"{source_tool}.{source_param}", f"{target_tool}.{target_param}", count))
        
        # 按计数降序排序
        all_deps.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n--- 参数依赖图中最常见的 {top_n} 个依赖关系 ---")
        for source, target, count in all_deps[:top_n]:
            print(f"{source} -> {target} (计数: {count})")
        print("---\n")
    
    def _is_param_match(self, param1: str, param2: str) -> bool:
        """
        检查两个参数名是否表示同一概念
        (简单版本，可在子类中扩展和优化)
        
        Args:
            param1: 第一个参数名
            param2: 第二个参数名
            
        Returns:
            是否匹配
        """
        # 基本匹配
        if param1 == param2:
            return True
            
        # 规范化参数名
        p1 = param1.lower().replace('_', '').replace('-', '')
        p2 = param2.lower().replace('_', '').replace('-', '')
        
        if p1 == p2:
            return True
            
        # 简单的常见模式
        # 例如 objName 和 object_name 可能表示同一概念
        return False  # 基础版本保守处理 
    
    def update_graph(self, current_sequence):
        """
        更新参数依赖图，添加当前序列的依赖关系
        
        Args:
            current_sequence: 当前序列的执行历史记录
        """
        if not current_sequence:
            return
            
        # 解析当前序列的执行历史
        execution_history = []
        for step in current_sequence:
            if step.get("type") != "act_ob_pair":
                continue
            
            action = step.get("action", {})
            observation = step.get("observation", {})
            
            tool_name = action.get("parsed_content", {}).get("tool_name")
            inputs = action.get("parsed_content", {}).get("inputs", {})
            outputs = observation.get("parsed_content", {}) if isinstance(observation, dict) else {}
            if outputs["status"] == "failure":
                print(f'[DEBUG ] detect fail in generate execution histoty: {action}')
                
            else:
                execution_history.append({
                    "tool_name": tool_name,
                    "inputs": inputs,
                    "outputs": outputs
                })

            for it in execution_history:
                if it["outputs"]["status"] == "failure":
                    print(f'[DEBUG ] wrong still: {it}')
            # print(f'[DEBUG ] append new history record: {tool_name} {inputs} {outputs}')
        
        # 构建依赖关系
        self.build_from_execution_history(execution_history, self.improved_value_match_func)

    def build_from_sequences(self, sequences: list, infer_output_func) -> None:
        """
        从轨迹序列数据构建参数依赖图。
        Args:
            sequences: 轨迹数据列表，每个元素为一个trajectory字典，结构参考react_baseline_log_20250427_195731.json
            infer_output_func: 用于解析observation的函数，签名为(tool_name, inputs, observation) -> outputs
        """
        if not sequences:
            print("[ParamDep] No sequences provided.")
            return
        total_deps = 0
        # --- TODO: 可以被优化这里，每次都遍历很慢
        for traj in sequences:
            steps = traj.get("steps", [])
            execution_history = []
            # for every step in the trajectory
            for step in steps:
                if step.get("type") != "act_ob_pair":
                    continue
                # --- get action and observation
                action = step.get("action", {})
                observation = step.get("observation", {})
                tool_name = None
                inputs = {}
                if isinstance(action, dict):
                    if action.get("parsed_content") and isinstance(action["parsed_content"], dict):
                        tool_name = action["parsed_content"].get("tool_name")
                        inputs = action["parsed_content"].get("inputs", {})
                    else:
                        # --- not find parsed_content, try to parse raw_content
                        print(f'[ParamDep] Warning: action parsed_content not found or not dict: {action}')
                        raw = action.get("raw_content", "")
                        import re, json as _json
                        m = re.match(r"(\w+) with Action Input: (\{.*\})", raw)
                        if m:
                            tool_name = m.group(1)
                            try:
                                inputs = _json.loads(m.group(2))
                            except Exception:
                                inputs = {}
                    if not tool_name:
                        continue
                else:
                    continue
                obs_content = observation.get("content", "") if isinstance(observation, dict) else str(observation)
                # --- get structured outputs
                outputs = infer_output_func(tool_name, inputs, obs_content)
                execution_history.append({
                    "tool_name": tool_name,
                    "inputs": inputs,
                    "outputs": outputs
                })
            before = len(getattr(self, "dependencies", {}))
            self.build_from_execution_history(execution_history, self.improved_value_match_func)
            after = len(getattr(self, "dependencies", {}))
            total_deps += (after - before)
        print(f"[ParamDep] Built dependency graph from {len(sequences)} sequences, total new deps: {total_deps}")


# --- 新增 ExecutionHistory 类定义 ---
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
    
    def clear(self):
        """清空所有历史记录"""
        self.history.clear()
        if self.debug:
            print("  History: Cleared all records")

    def reset(self):
        """重置历史（clear 的别名）"""
        self.clear()

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
    

if __name__ == "__main__":
    pass