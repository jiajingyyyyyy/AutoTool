import json
import time
import os
import sys
import re
import yaml
from datetime import datetime
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
from dotenv import load_dotenv
load_dotenv()
# --- 标准导入 ---
from agents.base_agent import BaseAgent
from common.registry import registry
from agentboard.prompts.ReactBaselineAgent.alfworld_sys_prompt import ALFWORLD_SYS_PROMPT
from agentboard.prompts.ReactBaselineAgent.tool_query_sys_prompt import TOOL_QUERY_SYSTEM_PROMPT
from agentboard.prompts.ReactBaselineAgent.scienceworld_sys_prompt import SCIENCEWORLD_SYS_PROMPT

# --- AutoTool / Memory 导入 (带回退逻辑) ---
from autool.utils import call_model
from autool.message import Message, Role
from autool.memory import TemporaryMemory
from autool.utils.parser.alfworld import parse_alfworld_action
from autool.utils.parser.tool_query import parse_tool_query
from autool.utils.parser.scienceworld import parse_scienceworld_action
from autool.utils.parser.check_action import check_tool_failure
from autool.tools.toolkit import Toolkit, get_tool_desc


# --- 惯性组件导入 ---
try:
    from autool.core.tool_predict.datastruct import ToolGraph
# # #     # print('[DEBUG INFO]: ToolGraph is inited when imported?')
    # 使用包含 record_action 的 ALFWorld 版本
    from autool.core.param_completion.domain.alfworld import AlfworldParamCompletion 
    from autool.core.param_completion.domain.tool_query import ToolQueryParamCompletion
    from autool.core.param_completion.domain.scienceworld import ScienceWorldParamCompletion
    # *******************************************************
    HAS_INERTIA_COMPONENTS = True
except ImportError as e:
    print(f"警告: 未找到惯性组件 (in my agent): {e}")
    HAS_INERTIA_COMPONENTS = False
    
# --- Logging Setup (Original) ---
log_dir = "logs"
if not os.path.exists(log_dir): os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"/console/agent_log_{time.strftime('%m%d_%H%M')}.log")

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)
    
@registry.register_agent("ReactInertiaAgent") 
class ReactInertiaAgent(BaseAgent): 
    def __init__(self,
                llm_model,
                task_type="",
                memory_size=100,
                examples=[],
                instruction="",
                init_prompt_path=None,
                system_message="You are a helpful assistant.",
                need_goal=True,
                check_actions="",
                check_inventory="inventory",
                use_parser=True,
                max_think_iters=3,
                max_steps=50,
                debug=True,
                record_mod=True,
                # --- 新增: 惯性配置 ---
                use_inertia=True, # 惯性功能的总开关
                inertia_threshold=0.2, # 触发参数填充的置信度阈值
                inertia_alpha=0.5, # 频率与语义得分的权重
                inertia_k=2, # 用于预测的历史窗口大小
                inertia_max=1, # 最长的连续惯性调用链
                inertia_fallback_hint=True, # 惯性失败时是否给 LLM 提示?
                tag="",
                ):
        super().__init__()
        # --- 核心属性 ---
        self.llm_model = llm_model
        self.task_type = task_type
        self.memory = TemporaryMemory()
        self.goal = None
        self.init_obs = None
        self.use_parser = use_parser
        self.debug = debug
        # --- 状态跟踪 ---
        self.steps = 0
        self.think_count = 0
        self.max_think_iters = max_think_iters
        self.max_steps = max_steps
        self.round_llm_duration = 0.0
        self.token_counts = {"input_tokens": 0, "output_tokens": 0,}
        self.round_cost = 0.0 # Placeholder for round cost (if needed)
        self.tag = tag
        self.record_mod = record_mod
        # --- 日志与历史 ---
        # self.tool_sequence = [] # 历史现在由 param_completion 管理
        log_dir = os.path.join(os.getcwd(), 'logs'); os.makedirs(log_dir, exist_ok=True)
        # 使用不同的日志文件名以区别于基线
        self.log_file = os.path.join(f'/data/agentboard/examples/{self.task_type}_{self.tag}/trajectories', f"{self.task_type}_inertia_{inertia_threshold}_{inertia_alpha}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        self.action_log = {
            "start_time": time.time(),
            "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_sequences": 0, "total_tool_calls": 0, "total_inertial_calling": 0, "sequences": [], "overhead": {} }
        self.current_sequence = {}

        # --- 统计跟踪变量 ---
        self.cumulative_time = 0.0
        self.start_time = time.time()
        self.end_time = None
        # --- overhead ---
        self.total_graph_search_time = 0.0
        self.total_SimSCE_time = 0.0
        self.total_intuition_embedding_time = 0.0
        self.total_path_embedding_time = 0.0
        self.total_similarity_cost_time = 0.0
        self.total_graph_construction_time = 0.0
        self.total_inertial_sense_overhead = 0.0
        self.total_parser_time = 0.0
        self.total_updating_time = 0.0
        self.total_param_filling_time = 0.0
        self.total_generate_action_time = 0.0
        self.total_llm_time = 0.0
        # --- Prompting 组件 ---
        self.check_actions_cmd = check_actions
        self.check_inventory_cmd = check_inventory
        self.examples = examples
        
        self.system_base = system_message
        self.instruction = instruction

        if init_prompt_path is not None:
            print(f"Loading initialization prompt from file: {init_prompt_path}")
            try:
                with open(init_prompt_path, 'r', encoding='utf-8') as f:
                    self.init_prompt_dict = json.load(f)
                self.examples = self.init_prompt_dict.get("examples", self.examples)
                self.system_base = self.init_prompt_dict.get("system_msg", self.system_base)
                self.instruction = self.init_prompt_dict.get("instruction", "")
                print("Prompt loaded successfully.")
            except Exception as e:
                print(f"Error loading prompt file {init_prompt_path}: {e}.")
        # Format examples for the prompt
        if isinstance(self.examples, list):
            self.examples_str = "\n---\n".join(self.examples)
        elif isinstance(self.examples, str):
            self.examples_str = self.examples
        else:
            self.examples_str = "[No examples provided]"

        # --- 初始化惯性组件 ---
        self.use_inertia = use_inertia and HAS_INERTIA_COMPONENTS
        self.inertia_threshold = inertia_threshold
        self.inertia_alpha = inertia_alpha
        self.inertia_k = inertia_k
        self.inertia_window = deque(maxlen=inertia_k+2) # 用于存储最近的 k 次调用
        self.inertia_max = inertia_max
        self.continuous_inertial_call_count = 0 # 用于跟踪连续惯性调用次数
        self.inertia_fallback_hint = inertia_fallback_hint
        self.inertia_count = 0 # 用于跟踪惯性调用次数
        self.inertial_step = [] # 用于跟踪惯性调用的步骤
        self.inertia_count_sum = 0 # 用于跟踪总的惯性调用次数
        self.param_filling_meta_data = None
        self.continue_invalid_tool_count = 0
        if self.use_inertia:
            try:
                print("Initial inertial module...")
                if not os.path.exists(self.log_file):
                    print(f"Build log file: {os.path.dirname(self.log_file)}")
                    os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                # action_description = '/data/agentboard/examples/tool_description/alfworld_action_description.json'
# # #                 # print(f'[DEBUG INFO]: action_description is: {action_description}, log_file is: {self.log_file}')
                # tool_description = '/data/agentboard/prompts/ReactInertiaAgent/tool_description.json'
                # 创建日志文件
                if not os.path.exists(self.log_file):
                    print(f"Build log file: {self.log_file}")
                    with open(self.log_file, 'w', encoding='utf-8') as f:
                        pass

                tool_desc_dir = os.getenv('TOOL_DESC_FILE', './tool_doc')
                # 根据任务类型拼接完整路径
                tool_description_path = os.path.join(
                    tool_desc_dir, 
                    f'{self.task_type}_tool_description.json'
                )
                print(f"Tool description is: {tool_description_path}")
                self.tool_graph = ToolGraph()
                # self.tool_graph.load_from_json(tool_description_path, '/data/logs/scienceworld_react_baseline_log_20250512_144931.json')
                self.tool_graph.load_from_json(tool_description_path, self.log_file)
                # --- 检查是否加载成功工具图 ---
# # #                 print('f[DEBUG INFO]: check if tool graph is loaded successfully')
                print(self.tool_graph.to_json()) 
                # baseline_trajectory = '/data/logs/react_baseline_log_20250417_092216.json'
                # self.tool_graph.load_from_json(action_description, baseline_trajectory)
                self._init_param_completion(tool_description_path)
                # 根据 agent 的 debug 标志设置 debug
                self.param_completion.set_debug(self.debug)
                self.param_dependency_path = os.path.join(f"/data/agentboard/examples/{self.task_type}/param_dependency_path/{self.task_type}", f"param_dependency_{time.strftime('%Y%m%d_%H%M%S')}.json")
                print("惯性组件初始化成功。")
            except Exception as e:
                print(f"错误：初始化惯性组件失败: {e}。正在禁用惯性功能。")
                self.use_inertia = False
        else:
            print("[ERROR]: 惯性组件不可用或已通过配置禁用。")
            self.tool_graph = None
            self.param_completion = None
            self.param_dependency_path = None

    def _init_param_completion(self, tool_description_path=None):
        try:
            # 【添加】读取 tool_descriptions
            with open(tool_description_path, 'r', encoding='utf-8') as f:
                tool_description = json.load(f)
            # print(tool_description)
            if self.task_type == 'alfworld':    
                self.param_completion = AlfworldParamCompletion(self.tool_graph,
                                                                tool_description,
                                                                dependency_graph_path=self.log_file,
                                                                history_max_len=20,
                                                                goal=self.goal,
                                                                debug=self.debug
                )
            elif self.task_type in ['tool_query', 'tool-query', 'tool-operation', 'academic']:
                print(f'[debug] start to init ToolQueryParamCompletion')
                self.param_completion = ToolQueryParamCompletion(self.tool_graph,
                                                                    tool_description,
                                                                    dependency_graph_path=self.log_file,
                                                                    history_max_len=20,
                                                                    goal=self.goal,
                                                                    debug=self.debug)
# # #                 print('f[DEBUG INFO]: ToolQueryParamCompletion is inited')
            elif self.task_type == 'scienceworld':
# # #                 print(f'[DEBUG INFO]: ScienceWorldParamCompletion is inited')
                self.param_completion = ScienceWorldParamCompletion(self.tool_graph,
                                                                    tool_description,
                                                                    dependency_graph_path=self.log_file,
                                                                    history_max_len=20,
                                                                    goal=self.goal,
                                                                    debug=self.debug)
        except Exception as e:
            print(f"错误：初始化参数填充组件失败: {e}。")
            self.param_completion = None

    def _parse_raw_action(self, action: str, observation: str) -> Optional[Dict[str, Any]]:
        """
        最后返回的统一格式
        parsed_action = {"tool_name": "unknown", "inputs": {}}
        """
        parsed_action_result = None
        if self.task_type == "alfworld":
            print(f"Debug: Parsing action: {action}")
            parsed_action_result = parse_alfworld_action(action, observation)
        elif self.task_type in ["tool_query", "tool-query", "academic"]: 
            parsed_action_result = parse_tool_query(action)
            print(f"Parsed Action: {parsed_action_result}")
        elif self.task_type == "scienceworld":
            parsed_action_result = parse_scienceworld_action(action)
            print(f"Parsed Action: {parsed_action_result}")
        
        return parsed_action_result


    def reset(self, goal, init_obs, init_act=None):
        """重置 agent 状态、内存、统计数据和惯性组件。"""


        if self.current_sequence:
            self._finalize_and_log_trajectory()

        self.memory.clear_memory() # 修正拼写错误
        self.param_completion.framework.history.reset() # 清除历史记录
        self.goal = goal
        self.init_obs = init_obs
        self.steps = 0
        self.think_count = 0
        # self.tool_sequence = [] # 历史由 param_completion 管理
        self.inertia_count = 0
        self.cumulative_time = 0.0

        # --- 重置惯性状态 ---
        if self.use_inertia and self.param_completion:
            print("正在重置惯性状态...")
            try:
                self.param_completion.adapter.reset()
                # self.param_completion.adapter.update_state(init_obs)
            except Exception as e:
                print(f"重置时更新场景知识出错: {e}")
            print("惯性状态已重置。")
        # ---
        start_time = datetime.now()
        

        self.current_sequence = {
            "trajectory_id": len(self.action_log.get("sequences", [])),
            "metadata": {
                "goal": self.goal,
                "start_time": start_time.strftime("%Y%m%d_%H%M%S.%f"),
                "end_time": None,
                "total_time_seconds": None,
                "status": "In Progress", # Initial status
                "total_llm_calls": 0, # Track LLM calls per trajectory
                "total_input_token": 0,
                "total_output_token": 0
            },
            "inertial_step": [], # Optional: For environment rewards
            "steps": [ # Store steps as structured dicts
                {"type": "initial",
                 "content": init_obs,
                 "timestamp": start_time.strftime("%Y%m%d_%H%M%S.%f")}
            ],
            # "run_stats": [], # To store stats from each agent.run() call
            "summary": None # Final summary (e.g., success/failure reason)
        }
        system_prompt_content = ""
        # 使用toolkit的get_tool_desc获取工具描述
        if hasattr(self, 'toolkit') and self.toolkit:
            tool_description = get_tool_desc(self.toolkit)
        else:
            # 如果没有toolkit，使用原来的instruction
            tool_description = self.instruction
            

        system_prompt_content = self._make_sys_prompt(tool_description=tool_description)
        # print(f"System prompt content: {system_prompt_content}")
        self.memory.add_memory(Message(Role.SYSTEM, system_prompt_content))
        self.memory.add_memory(Message(Role.USER, f"<Observation>{init_obs}</Observation>\n\n"))

    def parse_response_str(self, response_str):
        """Parse LLM response strictly for Think: and Action:"""
        if not isinstance(response_str, str): return None, None
        response_str = response_str.strip()

        # Improved regex to handle variations in spacing and case, and ensure Action: has content
        match = re.search(r"Think:\s*(.*?)\s*Action:\s*(.+)", response_str, re.DOTALL | re.IGNORECASE)

        if match:
            thought = match.group(1).strip()
            # Get potential action, strip extra lines/whitespace after it
            action = match.group(2).strip().split('\n')[0].strip()
            if action: # Ensure action is not empty
                return thought, action
            else:
                print(f"Warning: Parsing found 'Action:' but content is empty.")
                return thought, None # Return thought even if action is empty
        else:
            # Check if only Think: exists without a following Action:
            think_only_match = re.search(r"Think:\s*(.*)", response_str, re.DOTALL | re.IGNORECASE)
            if think_only_match:
                 thought = think_only_match.group(1).strip()
                 # Check if "Action:" appears *anywhere* after the thought
                 if "Action:" not in response_str[think_only_match.end():]:
                     print(f"Warning: Parsing found 'Think:' but no subsequent 'Action:'.")
                     return thought, None
                 else: # Action: exists but maybe format is wrong?
                      print(f"Warning: Parsing failed. Found 'Think:' and 'Action:' but not in expected sequence/format.")
                      return None, None # Indicate parsing failure if format is wrong
            else: # Neither Think: nor the specific pattern found
                print(f"Warning: Parsing failed. Could not find 'Think:...Action:...' structure.")
                return None, None
        
    def update(self, action='', state=''):
        self.steps += 1
        observation = state
        
        # --- 标准日志记录 ---
        if not isinstance(self.current_sequence, dict):
            print("Error: current_sequence not initialized correctly in update.")
            return
        
        parsed_action_result = None
        parsed_observation_result = None
        
        # --- 解析动作 ---
        if action and self.use_parser:
            s_time = time.time()
            parsed_action_result = self._parse_raw_action(action, observation)
            self.total_parser_time += time.time() - s_time
        
        # --- 解析 observation ---
        # 【修改】添加 param_completion 的空值检查
        if observation and self.use_parser and parsed_action_result:
            s_time = time.time()
            # 只在 param_completion 可用时才解析 observation
            if self.param_completion is not None:
                try:
                    parsed_observation_result = self.param_completion.adapter.infer_output(
                        parsed_action_result["tool_name"], 
                        parsed_action_result["inputs"], 
                        observation
                    )
                except Exception as e:
                    print(f"Warning: Failed to parse observation: {e}")
                    parsed_observation_result = None
            else:
                print("Warning: param_completion is None, skipping observation parsing")
                parsed_observation_result = None
            
            self.total_parser_time += time.time() - s_time

        # --- 解析 observation ---
        if observation and self.use_parser:
            s_time = time.time()
            parsed_observation_result = self.param_completion.adapter.infer_output(parsed_action_result["tool_name"], parsed_action_result["inputs"], observation)
            self.total_parser_time += time.time() - s_time
# # #             # print(f'[DEBUG INFO]: parsed_observation_result is: {parsed_observation_result}')
        if action is not None and observation is not None:
            type = "act_ob_pair"
            step_record = {
                "type": type,
                "step_id": self.steps-1,
                "param_filling_mode": self.param_filling_meta_data, # Placeholder for inertial data if needed
                "action": {
                    "raw_content": action,
                    "parsed_content": parsed_action_result, # Store parsed result (can be None or unknown)
                    "llm_time_cost": self.round_llm_duration
                },
                "observation": {
                    "raw_content": observation,
                    "parsed_content": parsed_observation_result, # Placeholder for parsed observation if needed
                },
                "token_consumption": self.token_counts.copy(), # Placeholder for token counts if implemented
                "round_time_cost": self.round_cost,
            }
            self.param_filling_meta_data = None
            print(f"Step {self.steps:02} - llm_time_cost: {self.round_llm_duration:.2f}s, round_cost: {self.round_cost:.2f}s")
            self.current_sequence.setdefault("steps", []).append(step_record)

            user_prompt_content = f"<Observation>{observation}</Observation>\n\n"
            self.memory.add_memory(Message(Role.USER, user_prompt_content))
        
        s_time = time.time()
        # --- 对于惯性调用加入反馈机制 ---
        if parsed_action_result.get("tool_name") != "unknown":
            self.inertia_window.append(parsed_action_result["tool_name"])
            print(f"Current inertia window: {self.inertia_window}")
            if self.round_llm_duration == 0.0:
                if check_tool_failure(observation, self.task_type):
                    print(f"Warning: 发现工具惯性调用失败，加入负权边 {self.inertia_window}")
                    self.tool_graph.record_tool_sequence(self.inertia_window, weight=-1)
                else:
                    print(f"Warning: 发现工具惯性调用成功，加入正权边 {self.inertia_window}")
                    self.tool_graph.record_tool_sequence(self.inertia_window, weight=1)
                
            else:
                if check_tool_failure(observation, self.task_type):
                    print(f"Warning: 发现工具LLM调用失败，加入负权边 {self.inertia_window}")
                    self.tool_graph.record_tool_sequence(self.inertia_window, weight=-0.5)
                else:
                    print(f"Warning: 发现工具LLM调用成功，加入正权边 {self.inertia_window}")
                    self.tool_graph.record_tool_sequence(self.inertia_window, weight=0.5)
        # --- 新增: 使用 ParamCompletion 记录动作 ---
        # 必须在每一步之后调用以维护历史记录
        if self.use_inertia and self.param_completion and action and state:
            try:
                # 这个方法会根据动作和观察更新 param_completion 内部的
                # executed_tools, tool_outputs, inventory, objects 等。
                if parsed_action_result and parsed_action_result.get("tool_name") != "unknown":
                    self.param_completion.framework.record_execution(action, parsed_action_result, state)
                    if self.debug: 
                        print(f"为惯性跟踪记录的动作: {action}")
            except Exception as e:
                print(f"错误: 调用 param_completion.record_action 时出错: {e}")
        # ---
        self.total_updating_time += time.time() - s_time

        print('*' * 20, "Inertail appended time cost display", '*' * 20)
        print(f"total_updating_time cost: {self.total_updating_time:.2f}s")
        print(f"total_graph_search_time cost: {self.total_graph_search_time:.2f}s")
        print(f"total_SimSCE_time cost: {self.total_SimSCE_time:.2f}s")
        print(f"total_intuition_embedding_time cost: {self.total_intuition_embedding_time:.2f}s")
        print(f"total_path_embedding_time cost: {self.total_path_embedding_time:.2f}s")
        print(f"total_similarity_cost_time cost: {self.total_similarity_cost_time:.2f}s")
        print(f"total_graph_construction_time cost: {self.total_graph_construction_time:.2f}s")
        print(f"total_inertial_sense_overhead cost: {self.total_inertial_sense_overhead:.2f}s")
        print(f"total_param_filling_time cost: {self.total_param_filling_time:.2f}s")
        print(f"total_generate_action_time cost: {self.total_generate_action_time:.2f}s")
        print(f"total_parser_time cost: {self.total_parser_time:.2f}s")
        print(f"total_llm_time cost: {self.total_llm_time:.2f}s")
        print('*' * 20, "Inertail appended time cost display", '*' * 20)


    def _make_sys_prompt(self, tool_description=""):
        """
        生成系统提示词
        """
        if self.task_type == 'alfworld':
            system_prompt_content = ALFWORLD_SYS_PROMPT.format(
                system_base=self.system_base, goal=self.goal,
                check_actions_cmd=self.check_actions_cmd, 
                check_inventory_cmd="inventory",
                examples_str=self.examples_str
            )
        elif self.task_type in ['tool_query', 'tool-query', 'tool-operation', 'academic']:
            system_prompt_content = TOOL_QUERY_SYSTEM_PROMPT.format(
                goal=self.goal,
                tools_description=tool_description,
                examples_str=self.examples
            )
        elif self.task_type == "scienceworld":
            system_prompt_content = SCIENCEWORLD_SYS_PROMPT.format(
                goal=self.goal,
                tools_description=tool_description,
                examples_str=self.examples
            )
        else:
            # Fallback to base template
            system_prompt_content = ALFWORLD_SYS_PROMPT.format(
                system_base=self.system_base, goal=self.goal,
                check_actions_cmd=self.check_actions_cmd, check_inventory_cmd=self.check_inventory_cmd,
                examples_str=self.examples_str
            )
        # print(f'[DEBUG] instr') # Debugging output
        # Add the tool description if available
        # print(f'[DEBUG] System Prompt: {system_prompt_content}.') # Debugging output
        return system_prompt_content



    # ------ Finalization and Logging Methods ------
    def _finalize_and_log_trajectory(self, status="Completed", summary="Trajectory finished."):
        """Finalizes the current trajectory log and appends it to the main log."""
        if not isinstance(self.current_sequence, dict):
            # print("Info: No active trajectory to finalize.")
            return

        end_time = datetime.now()
        start_time_str = self.current_sequence["metadata"]["start_time"]
        try:
            start_time = datetime.strptime(start_time_str, "%Y%m%d_%H%M%S.%f")
            total_time_seconds = (end_time - start_time).total_seconds()
        except ValueError:
            start_time = None
            total_time_seconds = self.cumulative_time # Use cumulative if start parse fails

        # Update metadata
        self.current_sequence["metadata"]["end_time"] = end_time.strftime("%Y%m%d_%H%M%S.%f")
        self.current_sequence["metadata"]["total_time_seconds"] = round(total_time_seconds, 3)
        self.current_sequence["metadata"]["status"] = status
        # Add cumulative stats to the trajectory log
        # self.current_sequence['final_cumulative_time_sec'] = round(self.cumulative_time, 3)
        # Add final token counts if tracked

        self.current_sequence["summary"] = summary
        # print(f"[debug in finalize_and_log_trajectory]: {self.current_sequence}")
        # Append the completed sequence to the main log list
        self.action_log["sequences"].append(self.current_sequence)
        self.action_log["total_sequences"] = len(self.action_log["sequences"])
        self.action_log["processed_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.action_log["inertial_step"] = self.inertial_step
        # self.action_log["total_task"] = len(self.action_log["sequences"])
        self.total_inertial_calling = len(self.inertial_step)
        self.action_log["total_inertial_calling"] = self.total_inertial_calling
        self.action_log["total_tool_calls"] = sum(len(seq.get("steps", [])) for seq in self.action_log["sequences"])
        self.action_log["overhead"] = {
            "total_graph_search_time": self.total_graph_search_time,
            "total_SimSCE_time": self.total_SimSCE_time,
            "total_intuition_embedding_time": self.total_intuition_embedding_time,
            "total_path_embedding_time": self.total_path_embedding_time,
            "total_similarity_cost_time": self.total_similarity_cost_time,
            "total_graph_construction": self.total_graph_construction_time,
            "total_inertial_sense_overhead": self.total_inertial_sense_overhead,
            "total_parser_time": self.total_parser_time,
            "total_updating_time": self.total_updating_time,
            "total_param_filling_time": self.total_param_filling_time,
            "total_generate_action_action_time": self.total_generate_action_time,
            "total_llm_time": self.total_llm_time,
        }
        

        # Save the entire log file
        self._save_action_log()
        # self.tool_graph.update_graph(self.current_sequence) # 更新图
        # print(json.dumps(self.tool_graph.to_json(), indent=2))

        s_time = time.time()
        # --- 更新 ToolGraph ---
        if self.use_inertia and self.tool_graph:
            try:
                print("Updating ToolGraph from the completed trajectory...")
                self.tool_graph.update_graph(self.current_sequence)
                print("ToolGraph updated.")
                # print(self.tool_graph.to_json()) 
            except Exception as e:
                 print(f"Error updating ToolGraph: {e}")

        # --- 更新并保存 ParameterDependencyGraph ---
        if self.use_inertia and self.param_completion:
            try:
                print(f"Updating ParameterDependencyGraph using current sequences...")
                # 使用当前内存中的序列数据构建依赖图
                self.param_completion.framework.param_graph.update_graph(self.current_sequence["steps"]) 
                # self.param_completion.framework.param_graph.build_from_sequences(self.action_log["sequences"], self.param_completion.adapter.infer_output)
                print(f"Saving updated ParameterDependencyGraph to: {self.param_dependency_path}...")
                self.param_completion.framework.param_graph.save_to_file(self.param_dependency_path)
                print("ParameterDependencyGraph updated and saved.")
            except Exception as e:
                 print(f"Error updating or saving ParameterDependencyGraph: {e}")
                 import traceback
                 traceback.print_exc()
        # ---

        self.total_graph_construction_time += time.time() - s_time
        # Clear current_sequence for the next run
        self.current_sequence = None

    def _save_action_log(self):
        """Saves the entire action log dictionary to the JSON file."""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.action_log, f, indent=2, ensure_ascii=False, cls=SetEncoder)
            # print(f"Action log saved to {self.log_file}")
        except Exception as e:
            print(f"Error saving log file {self.log_file}: {e}")


    def _check_param_filling_success(self, predicted_next_type: str, filled_params: Dict[str, Any]) -> bool:
        """辅助函数，检查是否所有必需的参数都已填充。"""
        # 特判
# # #         print(f"[DEBUG INFO]: predicted_next_type is: {predicted_next_type}, filled_params is: {filled_params}")
        if not self.use_inertia or not self.tool_graph: return False
        if predicted_next_type not in self.tool_graph.nodes: return False # 无法检查需求

        tool_node = self.tool_graph.nodes[predicted_next_type]
        required_param_names = set(tool_node.input_params.keys())
# # #         print("[DEBUG INFO]: required_param_names is: ", required_param_names)
        # 处理没有参数的工具 (如 'inventory', 'look')
        if not required_param_names:
            return True # 不需要参数，填充自然成功

        filled_param_names = set(filled_params.keys())

        missing = required_param_names - filled_param_names
        if not missing:
            if self.debug: print(f"参数填充检查: 成功 - {predicted_next_type}。已填充: {filled_params}")
            return True
        else:
            if self.debug: print(f"参数填充检查: 失败 - {predicted_next_type}。缺少: {missing}。已填充: {filled_params}")
            return False


    def run(self, init_prompt_dict=None):
        """运行 agent 一步，可能使用惯性调用。"""
        # print('=' * 50); print(f"GOAL: {self.goal}"); print(f"Current Step: {self.steps}"); print('=' * 50)

        run_start_time = time.time()
        llm_call_start_time = 0.0
        thought_count_this_step = 0
        self.round_llm_duration = 0.0
        self.round_cost = 0.0
        final_action = None
        llm_call_success = False # 标记 LLM 调用本身是否成功（返回非空）
        execute_inertially = False # 标记是否最终通过惯性执行跳过了 LLM
        inertia_prediction_made = False # 标记是否进行了预测尝试
        filling_successful = False # 标记参数填充是否成功 (仅当尝试惯性执行时)

        inertial_flag = True # 标记当前轮次是否考虑惯性
        

        # # --- 1. 尝试惯性预测和填充 (在 LLM 循环之前) ---
        if self.continuous_inertial_call_count >= self.inertia_max or self.inertia_count / (self.steps + 1) > 0.3:
            inertial_flag = False
            self.continuous_inertial_call_count = 0



        # --- 1.1 先看看有没有 nothing happen ，如果有直接调用 check ---
        if self.memory.get_memory(parse=True):
            last_memory = self.memory.get_memory(parse=True)[-1]["content"] 
            inertia_prediction_made = True
            if check_tool_failure(last_memory, self.task_type):
                self.continue_invalid_tool_count += 1    
                inertial_flag = False # 上一次出现无效调用，强制这次不使用惯性调用

            else: self.continue_invalid_tool_count = 0
            # --- predefined inertial chains 
            if self.continue_invalid_tool_count >= 2:
                print(f"--- 发现 ' 连续两次 Invalid action'，尝试调用 '{self.check_actions_cmd}' ---")

                final_action = self.check_actions_cmd
                # 固定模式触发，设置相关变量用于记录和展示
                execute_inertially = True
                bundled_think = f"Think: Detected 'Nothing happens' pattern. Applying action '{self.check_actions_cmd}' to recover."
                bundled_memory_msg = Message(Role.ASSISTANT, f"{bundled_think}\nAction: {final_action}")
                self.memory.add_memory(bundled_memory_msg)
                self.token_counts["input_tokens"] = 0
                self.token_counts["output_tokens"] = 0
                # 增加当前步骤的惯性计数
                self.inertia_count += 1 
                self.inertia_count_sum += 1
                if self.inertial_step and self.inertial_step[-1][1] == self.steps - 1:
                    self.continuous_inertial_call_count += 1
                self.inertial_step.append((len(self.action_log.get("sequences", [])), self.steps, final_action))
                self.current_sequence['inertial_step'].append((self.steps, final_action))
                return (final_action is not None), final_action
            
        # --- 1.2 检查是否启用并满足惯性预测条件 ---
        can_predict = (inertial_flag and
                       self.use_inertia and
                       self.param_completion and
                       self.tool_graph and
                       len(self.param_completion.framework.history.history) > self.inertia_k) # 使用 ExecutionHistory


        if can_predict:
            # 使用 param_completion 中的 history (结构化记录)
            # 注意：确保 history 存储的是 tool_name 列表或类似结构
            # 如果 history 存储的是完整字典，需要先提取 tool_name
            # 假设 param_completion 有一个方法或属性可以获取工具名历史
            # current_history_types = self.param_completion.get_tool_name_history(last_k=self.inertia_k)
            # 简化：直接从 ExecutionHistory 获取 tool_name
            recent_records = self.param_completion.framework.history.get_latest_records(k=self.inertia_k)
            print(f"--- 惯性检查: 最近 {len(recent_records)} 条历史记录: {recent_records} ---")
            recent_history_types = [
                rec.get("tool_name") 
                for rec in recent_records 
                if rec.get("tool_name") and rec.get("tool_name") != "unknown"  # 过滤 unknown
            ]
            if recent_history_types: # 需要有有效的历史工具名
                if self.debug: print(f"--- 惯性检查: 当前历史 (最近 {min(self.inertia_k, len(recent_history_types))} 步): {recent_history_types} ---")

                current_thought = self.memory.get_memory(parse=True)[-2]["content"]
                intuition = current_thought[-50:] if len(current_thought) > 50 else current_thought
                try:
                    s_time = time.time()
                    # 预测下一个动作类型
# # #                     # print(f'[DEBUG INFO]: start to predict next tool with chain similarity')
                    s_time = time.time()
                    has_inertia, inertial_sense_overhead, predicted_type = self.tool_graph.predict_next_tool_with_chain_similarity(
                        recent_history_types,
                        intuition=intuition,
                        threshold=self.inertia_threshold, # 在这里使用阈值判断是否高置信度
                        alpha=self.inertia_alpha
                    )
# # #                     print(f"[DEBUG INFO]: predicted_type is: {predicted_type}, has_inertia is: {has_inertia}")
                    self.total_inertial_sense_overhead = time.time() - s_time
                    inertia_prediction_made = True # 标记已进行预测尝试
                    self.total_graph_search_time += inertial_sense_overhead["search_time"]
                    self.total_SimSCE_time += inertial_sense_overhead["SimSCE_time"]
                    self.total_intuition_embedding_time += inertial_sense_overhead["intuition_embedding_time"]
                    self.total_path_embedding_time += inertial_sense_overhead["path_embedding_time"]
                    self.total_similarity_cost_time += inertial_sense_overhead["similarity_cost_time"]
        
                    if has_inertia and predicted_type: # 如果预测结果置信度高
                        print(f"--- 惯性预测: 高置信度预测为 '{predicted_type}' (得分 > {self.inertia_threshold}) ---")
                        print(f"--- 尝试为 '{predicted_type}' 进行参数填充 ---")

# # #                         # print(f"[DEBUG INFO]: test complete_params_with_inertial_chain")
                        # 尝试参数填充
                        s_time = time.time()
                        self.param_filling_meta_data, filled_params = self.param_completion.fill_parameters(
                            predicted_type, {}, self.inertia_k
                        )
# # #                         if self.debug: print("[DEBUG INFO]: filled_params is: ", filled_params)
                        # 检查填充是否成功
                        filling_successful = self._check_param_filling_success(predicted_type, filled_params)
                        self.total_param_filling_time += time.time() - s_time
                        if filling_successful:
                            print(f"--- 惯性成功: '{predicted_type}' 的参数已填充 ---")
                            # 生成动作字符串
                            s_time = time.time()
                            final_action = self.param_completion.adapter.generate_action_from_params(predicted_type, filled_params)
                            print(f"--- 执行惯性动作: {final_action} ---")
                            self.total_generate_action_time += time.time() - s_time
                            execute_inertially = True # 设置标志以跳过 LLM
                            self.inertia_count += 1 # 增加惯性调用计数
                            self.inertia_count_sum += 1 # 增加总的惯性调用计数
                            if self.inertial_step and self.inertial_step[-1][1] == self.steps - 1:
                                self.continuous_inertial_call_count += 1
                            self.inertial_step.append((len(self.action_log.get("sequences", [])), self.steps, final_action)) # 记录惯性调用的步骤
                            self.current_sequence['inertial_step'].append((self.steps, final_action)) # 记录惯性调用的步骤
                            # 注意：这里我们认为动作决策成功了，即使LLM没调用
                            # 添加图惯性执行的思考和动作到记忆
                            graph_think = f"Think: Using graph inertia to predict next action '{predicted_type}' with parameters {filled_params}."
                            graph_memory_msg = Message(Role.ASSISTANT, f"{graph_think}\nAction: {final_action}")
                            self.memory.add_memory(graph_memory_msg)
                            self.token_counts["input_tokens"] = 0
                            self.token_counts["output_tokens"] = 0
                            # 惯性成功，直接返回
                            return (final_action is not None), final_action
                        else:
                            print(f"--- 惯性回退: '{predicted_type}' 的参数填充失败。将继续执行 ReAct。---")
                            # 添加惯性调用记忆
                            inertial_mem = f"<InertialAction>Inertial action: {predicted_type} with params: {filled_params}</InertialAction>"
                            self.memory.add_memory(Message(Role.ASSISTANT, inertial_mem))
                    elif self.debug:
                        # 预测存在但置信度低，或根本没预测出来
                        if predicted_type:
                             print(f"--- 惯性检查: 低置信度预测 '{predicted_type}'。将继续执行 ReAct。---")
                        else:
                             print(f"--- 惯性检查: 未预测到动作类型。将继续执行 ReAct。---")

                except Exception as e:
                    print(f"错误: 惯性预测/填充过程中出错: {e}。将继续执行 ReAct。")
            elif self.debug:
                 print("--- 惯性检查: 尚无历史记录。将继续执行 ReAct。---")


        # --- 2. 标准 ReAct 循环 (如果不是通过惯性执行) ---
        if not execute_inertially:
            if not inertia_prediction_made: # 仅在未进行惯性检查时打印此消息
                print("--- 进入标准 ReAct 循环 (未进行惯性检查) ---")
            # else: 已在惯性检查部分打印回退消息

            while thought_count_this_step < self.max_think_iters:
                thought_count_this_step += 1
                # print(f"--- 思考迭代 {thought_count_this_step} ---")

                # Prepare messages for LLM
                llm_input_messages = self.memory.get_memory(parse=True)
                if not llm_input_messages:
                    print("Error: Memory is empty, cannot call LLM.")
                    break # Exit loop if memory is empty

                if self.debug:
                    print(" /// DEBUG: LLM Input ///")
                    print(f"System Prompt (start): {llm_input_messages[0]['content'][:200]}...")
                    print("--- History (Last 2 messages) ---")
                    for msg in llm_input_messages[-2:]: print(f"- {msg['role']}: {msg['content'][:200]}...")
                    print(" /// DEBUG END /// ")

                # --- Call LLM ---
                try:
                    # ** Replace with your actual LLM call **
                    llm_call_start_time = time.time()
                    # print(f'[IMPORTANT DEBUG]: llm_input_messages: {llm_input_messages}')
                    response_str, self.token_counts["input_tokens"], self.token_counts["output_tokens"] = call_model(llm_input_messages)
                    self.current_sequence["metadata"]["total_input_token"] += self.token_counts["input_tokens"]
                    self.current_sequence["metadata"]["total_output_token"] += self.token_counts["output_tokens"]
                    print(f'[debug after llm call]: {self.token_counts}')
                    self.round_llm_duration = time.time() - llm_call_start_time
                    self.total_llm_time += self.round_llm_duration
                    print(f"llm call time cost: {self.round_llm_duration:.2f}s")
                    llm_call_success = bool(response_str)
                    if llm_call_success:
                        self._increment_llm_calls() # Increment only on successful call
                    else: print("Error: LLM returned empty response.")
                except Exception as e:
                    print(f"LLM call failed: {e}")
                    llm_call_success = False
                    response_str = f"[LLM Call Error: {e}]" # Log the error


                if not llm_call_success:
                    # If LLM call fails on the first try, break immediately
                    if thought_count_this_step == 1: break
                    # If it fails on subsequent tries (e.g., correcting format), maybe try forcing help cmd
                    else: continue # Or break, depending on desired robustness

                if self.debug: print(f" /// DEBUG: LLM Raw Response ///\n{response_str}\n /// DEBUG END /// ")

                # Add LLM response to memory BEFORE parsing (important for context)
                self.memory.add_memory(Message(Role.ASSISTANT, response_str))

                # --- Parse Response ---
                thought, action = self.parse_response_str(response_str)

                if action: # Successfully parsed Think and Action
                    # print(f"Step {self.steps:02} (Parse OK) - Think: {thought}")
                    print('=' * 15, "  Action  ", '=' * 15); print(f"Action: {action}")
                    final_action = action
                    self.think_count = 0 # Reset consecutive think counter
                    parse_success = True
                    break # Exit loop, action decided
                # ---- 处理解析失败的情况 ----
                elif thought: # Parsed Think, but Action is missing/empty
                    print(f"Step {self.steps:02} (Action Missing/Empty) - Think: {thought}")
                    self.think_count += 1
                    if self.think_count >= self.max_think_iters:
                        print(f"Max think iters ({self.max_think_iters}) reached (Action missing). Forcing '{self.check_inventory_cmd}'.")
                        final_action = self.check_inventory_cmd
                        # Add a clarifying message to memory about the forced action
                        forced_response = f"Think: The previous response was missing an action after {self.max_think_iters} attempts. Forcing an inventory check.\nAction: {final_action}"
                        self.memory.add_memory(Message(Role.ASSISTANT, forced_response))
                        parse_success = True # We successfully forced an action
                        break
                    else:
                        # Ask LLM to provide the action based on its thought
                        user_prompt = f"<Observation>Your previous response correctly provided a thought but was missing the 'Action:'. Please provide the 'Action:' based on your thought: '{thought}'. Use the correct format: Think: [thought] Action: [action]</Observation>"
                        self.memory.add_memory(Message(Role.USER, user_prompt))
                        # Continue the loop to let LLM try again

                else: # Complete parsing failure (neither Think nor Action pattern found)
                    print(f"Step {self.steps:02} (Parse Failed)")
                    self.think_count += 1
                    if self.think_count >= self.max_think_iters:
                        print(f"Max think iters ({self.max_think_iters}) reached (Parsing failed). Forcing '{self.check_actions_cmd}'.")
                        final_action = self.check_actions_cmd
                        # Add a clarifying message to memory
                        forced_response = f"Think: The previous response format was incorrect after {self.max_think_iters} attempts. Forcing a check for valid actions.\nAction: {final_action}"
                        self.memory.add_memory(Message(Role.ASSISTANT, forced_response))
                        parse_success = True # We successfully forced an action
                        break
                    else:
                        # Ask LLM to correct the format
                        user_prompt = "<Observation>Your previous response format was incorrect. Please use the EXACT format: Think: [Your reasoning] Action: [Your single action]</Observation>"
                        self.memory.add_memory(Message(Role.USER, user_prompt))
            # --- 结束 ReAct 思考循环 ---

        # --- 3. 结束与统计 ---
        run_end_time = time.time()
        duration = run_end_time - run_start_time
        self.round_cost = duration # Store round cost for this run


        self.cumulative_time += duration

        return (final_action is not None), final_action


    def get_example_prompt(self): return self.examples_str

    def _increment_llm_calls(self):
        """Increment LLM call count for the current trajectory."""
        if isinstance(self.current_sequence, dict) and "metadata" in self.current_sequence:
            self.current_sequence["metadata"]["total_llm_calls"] += 1

    @classmethod
    def from_config(cls, llm_model, config):
        """从配置创建 agent。"""
        return cls(
                    llm_model=llm_model, # LLM model is usually passed separately
                    task_type=config.get("task_type", "alfworld"), # For future use, maybe for different environments
                    memory_size=config.get("memory_size", 100),
                    examples=config.get("examples", []),
                    instruction=config.get("instruction", ""), # May not be used directly by this prompt structure
                    init_prompt_path=config.get("init_prompt_path", None),
                    system_message=config.get("system_message", "You are a helpful assistant."),
                    need_goal=config.get("need_goal", True), # Should likely remain True
                    check_actions=config.get("check_actions", "check valid actions"),# check_valid_actions with Action Input: {}
                    check_inventory=config.get("check_inventory", "inventory"),
                    use_parser=config.get("use_parser", True),
                    max_think_iters=config.get("max_think_iters", 3),
                    max_steps=config.get("max_steps", 50),
                    debug=config.get("debug", True),
                    record_mod=config.get("record_mod", True),
                    # --- 传递惯性配置 ---
                    use_inertia=config.get("use_inertia", True),
                    inertia_threshold=config.get("inertia_threshold", 0.2),# 0.15
                    inertia_alpha=config.get("inertia_alpha", 0.5),
                    inertia_k=config.get("inertia_k", 2),
                    inertia_max=config.get("inertia_max", 1),
                    inertia_fallback_hint=config.get("inertia_fallback_hint", True),
                    tag=config.get("tag", "quickstart"),
                    )