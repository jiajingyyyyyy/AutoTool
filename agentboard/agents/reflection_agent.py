import json
import time
import os
import re
import re
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
# --- 标准导入 ---
from agents.base_agent import BaseAgent
from common.registry import registry
from agentboard.prompts.ReflectionBaselineAgent.academic_sys_prompt import TOOL_QUERY_REFLECTION_EXAMPLES, TOOL_QUERY_REFLECTION_PROMPT
from agentboard.prompts.ReflectionBaselineAgent.alfworld_sys_prompt import ALFWORLD_REFLECTION_EXAMPLES, ALFWORLD_REFLECTION_PROMPT
from agentboard.prompts.ReflectionBaselineAgent.scienceworld_sys_prompt import SCIENCEWORLD_REFLECTION_EXAMPLES, SCIENCEWORLD_REFLECTION_PROMPT
from agentboard.prompts.ReflectionBaselineAgent.reflection_instruction import REFLECTION_INSTRUCTIONS

def get_reflection_instructions(task_type, reflections=None):
    """根据任务类型和历史反思生成反思指令"""
    # 选择合适的反思示例
    if task_type.startswith("alfworld"):
        examples = ALFWORLD_REFLECTION_EXAMPLES
    elif task_type.startswith("scienceworld"):
        examples = SCIENCEWORLD_REFLECTION_EXAMPLES
    elif task_type.startswith("tool") or task_type.startswith("weather") or task_type.startswith("movie"):
        examples = TOOL_QUERY_REFLECTION_EXAMPLES
    else:
        examples = ""
    
    # 格式化已有的反思（如果有）
    formatted_reflections = ""
    if reflections and len(reflections) > 0:
        for i, reflection in enumerate(reflections[-3:]):  # 最多展示最近3条反思
            formatted_reflections += f"Reflection {i+1}: {reflection}\n"
    
    # 如果没有历史反思，提供一些示例
    if not formatted_reflections:
        reflection_text = f"""
**Reflection Examples:**
{examples}

Remember to learn from these examples when you encounter similar situations. Be adaptive and try different approaches if your current strategy isn't working.
"""
    else:
        reflection_text = REFLECTION_INSTRUCTIONS.format(reflections=formatted_reflections)
    
    return reflection_text 


# from agentboard.prompts.Reflection.reflection_instructions import get_reflection_instructions

# --- AutoTool / Memory 导入 (带回退逻辑) ---

from autool.utils import call_model
from autool.message import Message, Role
from autool.memory import TemporaryMemory
from autool.utils.parser.alfworld import parse_alfworld_action
from autool.utils.parser.tool_query import parse_tool_query
from autool.utils.parser.scienceworld import parse_scienceworld_action
from autool.tools.toolkit import Toolkit, get_tool_desc

HAS_AutoTool=True

    

@registry.register_agent("ReflectionAgent")
class ReflectionAgent(BaseAgent):
    def __init__(self,
                 llm_model,               # LLM模型名称 
                 task_type="alfworld",    # 任务类型
                 memory_size=100,         # 记忆大小
                 examples=[],             # 示例
                 instruction="",          # 指令
                 init_prompt_path=None,   # 初始化提示路径
                 system_message="You are a helpful assistant.",
                 check_actions="check valid actions",
                 check_inventory="inventory",
                 max_think_iters=3,       # 最大思考迭代次数
                 reflection_threshold=3,  # 反思阈值
                 few_shot_examples_path="./alfworld_runs/reflexion_few_shot_examples.txt",
                 prompts_json_path="./alfworld_runs/prompts/alfworld_3prompts.json",
                 debug=False,
                 record_mod=True,
                 tag="reflection_agent",
                 enable_reflection=True,  # 是否启用反思功能
                 ):
        super().__init__()

        # --- 核心属性 ---
        self.llm_model = llm_model
        self.task_type = task_type
        self.memory = TemporaryMemory() if HAS_AutoTool else TemporaryMemory()
        self.goal = None
        self.init_obs = None
        self.debug = debug
        self.examples = examples  # 保留参数，但不会直接使用
        self.tag = tag
        self.enable_reflection = enable_reflection  # 反思启用开关
        
        # --- 提示相关 ---
        self.system_base = system_message
        self.instruction = instruction
        self.check_actions_cmd = check_actions
        self.check_inventory_cmd = check_inventory
        
        # --- 状态跟踪 ---
        self.steps = 0
        self.think_count = 0
        self.max_think_iters = max_think_iters
        self.memory_size = memory_size
        self.round_llm_duration = 0.0
        self.round_cost = 0.0
        self.token_counts = {"input_tokens": 0, "output_tokens": 0}
        self.record_mod = record_mod
        self.cumulative_time = 0.0

        # --- Reflexion相关属性 ---
        self.reflection_threshold = reflection_threshold  # Now used for consecutive failures
        self.consecutive_same_action_count = 0
        self.last_action = ""
        self.reflections = []
        self.interaction_history = []
        self.reflection_count = 0  # 反思计数器
        self.reflection_timestamps = []  # 记录反思发生的时间点
        self.last_action_successful = True # Assume success initially
        self.consecutive_failures = 0      # Counter for consecutive failures
        
        # --- 任务类型和示例管理 ---
        self.task_prefix = None  # 任务类型前缀，用于选择合适的示例
        self.prompts_dict = {}   # 存储各种任务类型的示例
        print(f"init_prompt_path: {init_prompt_path}")
        if init_prompt_path is not None:
            print(f"Loading initialization prompt from file: {init_prompt_path}")
            try:
                with open(init_prompt_path, 'r', encoding='utf-8') as f:
                    self.init_prompt_dict = json.load(f)
                self.examples = self.init_prompt_dict.get("examples", self.examples)
                # print(f"Examples loaded: {self.examples}")
                self.system_base = self.init_prompt_dict.get("system_msg", self.system_base)
                self.instruction = self.init_prompt_dict.get("instruction", "")
                print("Prompt loaded successfully.")
            except Exception as e:
                print(f"Error loading prompt file {init_prompt_path}: {e}.")
        # --- 系统提示词导入 ---
        try:
            self.sys_prompts = {
                "alfworld": ALFWORLD_REFLECTION_PROMPT,
                "scienceworld": SCIENCEWORLD_REFLECTION_PROMPT,
                "tool_query": TOOL_QUERY_REFLECTION_PROMPT,
            }
            self.get_reflection_instructions = get_reflection_instructions
            print(f"Successfully loaded system prompts for multiple datasets")
        except ImportError as e:
            print(f"Warning: Could not import reflection system prompts: {e}")
            # 保留原有ALFWorld提示词格式作为备份
            self.sys_prompts = {}
            self.get_reflection_instructions = lambda task_type, reflections: "\n".join([f"Reflection {i+1}: {r}" for i, r in enumerate(reflections[-3:])] if reflections else "")
        
        # --- 日志 ---
        self.tool_sequence = []
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{self.task_type}_{self.tag}_log_{time.strftime('%Y%m%d_%H%M%S')}.json")
        self.action_log = {
            "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_sequences": 0,
            "sequences": []
        }
        self.current_sequence = None

        # 加载ALFWorld任务示例
        try:
            with open(prompts_json_path, 'r') as f:
                self.prompts_dict = json.load(f)
                print(f"Loaded {len(self.prompts_dict)} prompt examples from {prompts_json_path}")
        except Exception as e:
            print(f"Failed to load prompts from {prompts_json_path}: {e}")
            self.prompts_dict = {}

        # 加载few-shot反思示例
        try:
            with open(few_shot_examples_path, 'r') as f:
                self.few_shot_examples = f.read()
                print(f"Loaded few-shot reflection examples from {few_shot_examples_path}")
        except Exception as e:
            self.few_shot_examples = ""
            print(f"Warning: Could not load few-shot examples from {few_shot_examples_path}: {e}")
            
        # 打印Agent类型和配置信息
        self._print_agent_config()
        
    def _print_agent_config(self):
        """打印Agent配置信息，用于调试和区分不同类型的Agent"""
        agent_type = "ReflectionAgent" if self.enable_reflection else "ReactAgent"
        reflection_status = "ENABLED" if self.enable_reflection else "DISABLED"
        
        config_info = f"""
{'='*50}
Agent Type: {agent_type} (tag: {self.tag})
Task Type: {self.task_type}
Reflection: {reflection_status} (threshold: {self.reflection_threshold})
Model: {self.llm_model}
Debug Mode: {"ON" if self.debug else "OFF"}
{'='*50}
"""
        print(config_info)

    def _determine_task_prefix(self, goal):
        """从目标文本确定任务类型前缀"""
        goal_lower = goal.lower()
        
        # 任务类型到前缀的映射
        prefix_map = {
            'put': ['put', 'find', 'place'],
            'clean': ['clean'],
            'heat': ['heat', 'hot'],
            'cool': ['cool'],
            'puttwo': ['put two', 'place two'],
            'examine': ['look at', 'examine', 'check']
        }
        
        # 查找匹配的前缀
        for prefix, keywords in prefix_map.items():
            for keyword in keywords:
                if keyword in goal_lower:
                    return prefix
        
        # 默认为"put"
        return "put"

    def _evaluate_outcome(self, observation: str, action: str) -> bool:
        """评估上一个动作是否成功
        
        Args:
            observation: 当前观察
            action: 上一个执行的动作

        Returns:
            bool: 如果动作被认为是失败的，则返回True，否则返回False。
        """
        # 这是一个简化的评估器，实际应用中需要更复杂的逻辑
        # 例如，检查特定的失败关键词或无意义的观察结果
        # ALFWorld specific failure indicators
        if self.task_type == "alfworld":
            # Common failure phrases in Alfworld
            failure_keywords = [
                "nothing happens",
                "you don't see that",
                "it's not clear what you mean",
                "you can't",
                "is not", # e.g. "The drawer is not openable"
                "is closed", # often implies failure if an open action was attempted on something already closed or unopenable
                "already", # e.g. "The microwave is already on/off."
            ]
            # Observations that are too short or generic might also indicate failure or no change
            if len(observation) < 30 and "you see nothing special" in observation.lower():
                return True # Failed
            for keyword in failure_keywords:
                if keyword in observation.lower():
                    # Specific check for toggling devices that are already in the target state
                    if "already" in observation.lower() and action.startswith("toggle"):
                         print(f"Evaluator: Detected toggle on already set device for action '{action}'. Observation: '{observation}'")
                         return True # Failed - tried to toggle something already in state
                    elif not "already" in observation.lower(): # Avoid false positives for "already"
                        print(f"Evaluator: Detected failure keyword '{keyword}' for action '{action}'. Observation: '{observation}'")
                        return True # Failed

        elif self.task_type == "scienceworld":
            failure_keywords = [
                "i'm not sure what you mean by that",
                "that doesn't make sense",
                "you can't do that",
                "it's not possible to",
                "nothing happens",
                "no effect",
                "already focused on"
            ]
            if any(keyword in observation.lower() for keyword in failure_keywords):
                print(f"Evaluator: Detected failure keyword for action '{action}'. Observation: '{observation}'")
                return True # Failed
            # Check if observation is identical to previous, indicating no change
            if len(self.interaction_history) > 0:
                prev_obs = self.interaction_history[-1].get('observation', '')
                if observation.strip() == prev_obs.strip() and not action.lower().startswith("look") and not action.lower().startswith("inventory") and not action.lower().startswith("task"):
                    print(f"Evaluator: Observation unchanged after action '{action}'. Assuming failure.")
                    return True # Failed
        
        # Add more task-specific heuristics if needed for other task_types

        return False # Assumed successful if no failure indicators found

    
    def _make_sys_prompt(self):
        """根据任务类型创建相应的系统提示"""
        # 获取任务相关的示例，始终使用从JSON文件加载的示例
        # task_examples = self._get_task_examples()
        task_examples = self.examples
        # 准备反思指令
        try:
            reflection_instructions = self.get_reflection_instructions(
                task_type=self.task_type, 
                reflections=self.reflections
            )
        except Exception as e:
            print(f"Error generating reflection instructions: {e}")
            # 备用反思指令格式
            reflection_instructions = ""
            if self.reflections:
                reflection_instructions = "\n**Past Reflections:**\n"
                for i, reflection in enumerate(self.reflections[-3:]):
                    reflection_instructions += f"Reflection {i+1}: {reflection}\n"
        
        # 选择合适的系统提示词模板
        system_template = None
#         print(f'[DEBUG INFO]: task_type: {self.task_type}')
        if self.task_type == 'alfworld':
            system_template = self.sys_prompts["alfworld"]
        elif self.task_type == "scienceworld":
            system_template = self.sys_prompts["scienceworld"]
        elif self.task_type in ['tool_query', 'tool-query', 'tool-operation']:
            system_template = self.sys_prompts["tool_query"]
        
        # 如果没有找到匹配的模板，使用默认ALFWorld模板
        if not system_template:
            print(f"No system prompt template found for task type: {self.task_type}, using default ALFWorld template")
            # 构建系统提示
            system_prompt_content = f"""Your MAIN GOAL is: **{self.goal}**
You MUST achieve ALL aspects of this goal. Do not simplify or ignore parts of the goal.

Follow the ReAct (Reasoning + Acting) paradigm:
1.  **Think:**
    What you think about the current situation and how to achieve the goal.
2.  **Action:** Output EXACTLY ONE command from the list below based on your thought process.

Available Actions:
- **take {{obj}} from {{recep}}**: Pick up an object ('{{obj}}') from a location ('{{recep}}').
- **put {{obj}} in/on {{recep}}**: Place an object ('{{obj}}') into/onto a location ('{{recep}}').
- **open {{recep}}**: Open a container ('{{recep}}') to access its contents.
- **close {{recep}}**: Close an open container ('{{recep}}').
- **toggle {{target}}**: Toggle the state of an object or device ('{{target}}') (e.g., turn on/off a desklamp or microwave).
- **clean {{obj}} with {{recep}}**: Clean an object ('{{obj}}') with a tool/receptacle ('{{recep}}').
- **cool {{obj}} with {{recep}}**: Cool an object ('{{obj}}') with a cooling device/receptacle ('{{recep}}') like a fridge.
- **heat {{obj}} with {{recep}}**: Heat an object ('{{obj}}') with a heating device/receptacle ('{{recep}}') like a microwave.
- **examine {{target}}**: Look closely at an object or receptacle ('{{target}}') to get details. Use this to fulfill 'look at [object]' parts of a goal.
- **go to {{recep}}**: Move to a specific location or receptacle ('{{recep}}').
- **look**: Observe your current surroundings to get a general view of the area and visible items.
- **use {{target}}**: Use or interact with an object or device ('{{target}}') in its default way (often similar to 'toggle' for devices).
- **{self.check_inventory_cmd}**: Check what objects you are currently carrying.
- **{self.check_actions_cmd}**: List all currently valid actions based on the environment's state. (Use if unsure or if standard actions fail).

Guidelines for Action:

**CRITICAL FORMATTING:** Your response MUST be structured EXACTLY as follows, with "Think:" and "Action:" on SEPARATE lines and NO extra text before or after:
Think: [Your reasoning and plan here]
Action: [Your single, valid Alfworld command here from 'Available Actions']
"""

            # 仅当有任务相关示例时才添加示例部分
            if task_examples:
                system_prompt_content += f"""
**Interaction Examples:**
---
{task_examples}
---
"""
            
            # 添加反思指令
            if reflection_instructions:
                system_prompt_content += f"\n{reflection_instructions}\n"
            
            
            return system_prompt_content
            
        # 使用选定的模板格式化系统提示
        try:
            # 准备格式化参数
            format_args = {
                "goal": self.goal,
                "check_inventory_cmd": self.check_inventory_cmd,
                "check_actions_cmd": self.check_actions_cmd,
                "reflection_instructions": reflection_instructions,
                "examples_str": self.examples,
            }
            
            # 对于特定任务类型，添加额外的参数
            # if self.task_type.startswith("tool"):
            #     format_args["tools_description"] = self.instruction
            #     if task_examples:
            #         format_args["examples_str"] = task_examples
            # elif task_examples:
            #     format_args["examples_str"] = task_examples
            
            # 格式化模板
            system_prompt_content = system_template.format(**format_args)
            return system_prompt_content
            
        except Exception as e:
            print(f"Error formatting system prompt: {e}")
            # 出错时回退到简单格式
            return f"Your goal is: {self.goal}\n\n{reflection_instructions}"

    def reset(self, goal, init_obs):
        """
        reset() is called at the beginning of each episode.
        
        You can use this function to reset your agent's internal state.
        Args:
            goal: The goal string.
            init_obs: The initial observation string.
        
        """
        # 重置状态计数器
        start_time = datetime.now()
        
        # 如果有前一个序列，保存它
        if self.current_sequence:
            self._finalize_and_log_trajectory()
            
        # 重置内部状态
        self.memory.clear_memory()
        self.goal = goal
        self.init_obs = init_obs
        self.steps = 0
        self.think_count = 0
        self.tool_sequence = []
        self.cumulative_time = 0.0
        self.reflections = []
        self.interaction_history = []
        self.consecutive_same_action_count = 0
        self.last_action = ""
        
        # 确定任务前缀，用于选择合适的示例
        self.task_prefix = self._determine_task_prefix(goal)
        print(f"Task prefix determined: {self.task_prefix}")
        
        # 开始新的序列日志
        self.current_sequence = {
            "trajectory_id": len(self.action_log['sequences']),
            "metadata": {
                "goal": self.goal,
                "task_prefix": self.task_prefix,
                "start_time": start_time.strftime("%Y%m%d_%H%M%S.%f"),
                "end_time": None,
                "total_time_seconds": None,
                "status": "In Progress",
                "total_llm_calls": 0,
                "total_input_token": 0,
                "total_output_token": 0
            },
            "steps": [
                {"type": "initial",
                 "content": init_obs,
                 "timestamp": start_time.strftime("%Y%m%d_%H%M%S.%f")}
            ],
            "reflections": [],
            "rewards": [],
            "summary": None
        }
        
        # 设置初始系统提示
        system_prompt_content = self._make_sys_prompt()
#         print(f'[DEBUG INFO]: System prompt content: {system_prompt_content}')
        # 添加到记忆中
        if HAS_AutoTool:
            self.memory.add_memory(Message(Role.SYSTEM, system_prompt_content))
            self.memory.add_memory(Message(Role.USER, f"<Observation>{init_obs}</Observation>\n\n"))
        else:
            self.memory.add_memory(Message(Role.SYSTEM, system_prompt_content))
            self.memory.add_memory(Message(Role.USER, f"<Observation>{init_obs}</Observation>\n\n"))
        
        print(f"Agent Reset. Goal: {self.goal}")
        print(f"Initial Observation added to memory.")

    def run(self, init_prompt_dict=""):
        """
        run() is called at each time step to generate an action based on the agent current state.
        
        """
        # print(f'[DEBUG] init_prompt_dict: {init_prompt_dict}')
        print('=' * 50)
        print(f"GOAL: {self.goal}")
        print(f"Current Step: {self.steps}")
        # 如果启用了反思，显示反思统计信息
        if self.enable_reflection:
            print(f"Reflection Stats: {self.reflection_count} generated | " +
                  f"Consecutive Same Action: {self.consecutive_same_action_count}/{self.reflection_threshold} | " +
                  f"Consecutive Failures: {self.consecutive_failures}/{self.reflection_threshold}")
        print('=' * 50)

        run_start_time = time.time()
        thought_count_this_step = 0
        final_action = None
        llm_call_success = False
        parse_success = False
        response_str = ""
        is_reflection_step = False  # 标记当前步骤是否生成了反思

        # 检查是否需要生成反思 (基于连续失败)
        if self.enable_reflection and self.consecutive_failures >= self.reflection_threshold:
            is_reflection_step = True
            print(f"\n{'*'*20} GENERATING REFLECTION DUE TO {self.consecutive_failures} CONSECUTIVE FAILURES {'*'*20}")
            
            reflection = self._generate_reflection()
            if reflection:
                self.reflection_count += 1
                self.reflections.append(reflection)
                self.reflection_timestamps.append(self.steps)
                self.current_sequence.setdefault("reflections", []).append({
                    "step": self.steps,
                    "content": reflection,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S.%f")
                })
                
                # 显示生成的反思
                print(f"Reflection #{self.reflection_count}: {reflection}")
            self.consecutive_failures = 0 # 重置失败计数器
            self.consecutive_same_action_count = 0 # 也重置重复动作计数，因为反思会改变策略
            
            # 更新系统提示以包含新的反思
            self._update_system_prompt()
            print(f"{'*'*20} REFLECTION COMPLETE {'*'*20}\n")

        # --- ReAct思考循环 ---
        while thought_count_this_step < self.max_think_iters:
            thought_count_this_step += 1
            
            # 准备LLM的消息
            llm_input_messages = self.memory.get_memory(parse=True)
            if not llm_input_messages:
                print("Error: Memory is empty, cannot call LLM.")
                break
                
            if self.debug:
                print("\n/// DEBUG: LLM Input ///")
                print(f"System Prompt (start): {llm_input_messages[0]['content'][:200]}...")
                # 检查系统提示中是否包含反思
                if "reflection" in llm_input_messages[0]['content'].lower():
                    reflection_section = llm_input_messages[0]['content'].split("**Past Reflections")
                    if len(reflection_section) > 1:
                        print(f"Reflection Section: **Past Reflections{reflection_section[1].split('**')[0]}**")
                print("--- History (Last 2 messages) ---")
                for msg in llm_input_messages[-2:]:
                    print(f"- {msg['role']}: {msg['content'][:200]}...")
                print("/// DEBUG END /// ")
                
            # --- 调用LLM ---
            try:
                llm_call_start_time = time.time()
                if HAS_AutoTool:
                    # print(f'[IMPORTANT DEBUG]: llm_input_messages: {llm_input_messages}')
                    response_str, self.token_counts["input_tokens"], self.token_counts["output_tokens"] = call_model(llm_input_messages)
                    self.current_sequence["metadata"]["total_input_token"] += self.token_counts["input_tokens"]
                    self.current_sequence["metadata"]["total_output_token"] += self.token_counts["output_tokens"]
                else:
                    print('Could not call LLM: AutoTool not available')
                    
                self.round_llm_duration = time.time() - llm_call_start_time
                llm_call_success = bool(response_str)
                if llm_call_success:
                    self._increment_llm_calls()
                else:
                    print("Error: LLM returned empty response.")
            except Exception as e:
                print(f"LLM call failed: {e}")
                llm_call_success = False
                response_str = f"[LLM Call Error: {e}]"
                
            if not llm_call_success:
                if thought_count_this_step == 1:
                    break
                else:
                    continue
                    
            # 添加LLM响应到记忆
            if HAS_AutoTool:
                self.memory.add_memory(Message(Role.ASSISTANT, response_str))
            else:
                self.memory.add_memory(Message(Role.ASSISTANT, response_str))
                
            # --- 解析响应 ---
            thought, action = self.parse_response_str(response_str)
            
            if action:  # 成功解析Think和Action
                # 判断思考中是否提到了反思
                reflection_influenced = False
                if thought and any(r.lower() in thought.lower() for r in ["reflect", "reflection", "previous attempt", "past failure", "try different"]):
                    reflection_influenced = True
                    
                self._print_parsed_response(self.steps, thought, action, reflection_influenced, is_reflection_step)
                
                final_action = action
                self.think_count = 0  # 重置连续思考计数器
                parse_success = True
                
                # 跟踪重复动作
                if action == self.last_action:
                    self.consecutive_same_action_count += 1
                    if self.consecutive_same_action_count > 1:
                        print(f"⚠️ Warning: Same action repeated {self.consecutive_same_action_count} times")
                else:
                    self.consecutive_same_action_count = 0
                self.last_action = action
                
                break  # 退出循环，动作已决定
                
            elif thought:  # 解析到Think，但Action缺失/为空
                print(f"Step {self.steps:02} (Action Missing/Empty) - Think: {thought}")
                self.think_count += 1
                if self.think_count >= self.max_think_iters:
                    print(f"Max think iters ({self.max_think_iters}) reached (Action missing). Forcing '{self.check_inventory_cmd}'.")
                    final_action = self.check_inventory_cmd
                    forced_response = f"Think: The previous response was missing an action after {self.max_think_iters} attempts. Forcing an inventory check.\nAction: {final_action}"
                    if HAS_AutoTool:
                        self.memory.add_memory(Message(Role.ASSISTANT, forced_response))
                    else:
                        self.memory.add_memory(Message(Role.ASSISTANT, forced_response))
                    parse_success = True
                    break
                else:
                    user_prompt = f"<Observation>Your previous response correctly provided a thought but was missing the 'Action:'. Please provide the 'Action:' based on your thought: '{thought}'. Your response MUST be formatted EXACTLY as follows:\nThink: [your thought]\nAction: [your action]\nDo not include any other text.</Observation>"
                    if HAS_AutoTool:
                        self.memory.add_memory(Message(Role.USER, user_prompt))
                    else:
                        self.memory.add_memory(Message(Role.USER, user_prompt))
                        
            else:  # 完全解析失败（找不到Think或Action模式）
                print(f"Step {self.steps:02} (Parse Failed)")
                self.think_count += 1
                if self.think_count >= self.max_think_iters:
                    print(f"Max think iters ({self.max_think_iters}) reached (Parsing failed). Forcing '{self.check_actions_cmd}'.")
                    final_action = self.check_actions_cmd
                    forced_response = f"Think: The previous response format was incorrect after {self.max_think_iters} attempts. Forcing a check for valid actions.\nAction: {final_action}"
                    if HAS_AutoTool:
                        self.memory.add_memory(Message(Role.ASSISTANT, forced_response))
                    else:
                        self.memory.add_memory(Message(Role.ASSISTANT, forced_response))
                    parse_success = True
                    break
                else:
                    user_prompt = "<Observation>Your previous response format was incorrect. Your response MUST be formatted EXACTLY as follows:\nThink: [your detailed reasoning]\nAction: [your exact action command]\n\nThe 'Think:' and 'Action:' MUST be on separate lines with no additional text. Do not include any explanations or notes outside of these two lines.</Observation>"
                    if HAS_AutoTool:
                        self.memory.add_memory(Message(Role.USER, user_prompt))
                    else:
                        self.memory.add_memory(Message(Role.USER, user_prompt))
        
        # --- 思考循环结束 ---
        run_end_time = time.time()
        duration = run_end_time - run_start_time
        self.round_cost = duration
        
        # 更新累计统计信息
        self.cumulative_time += duration
        
        print('=' * 17, '  Run End  ', '=' * 17)
        print(f"Run Time: {duration:.2f}s")
        if final_action:
            print(f"Action Decided: {final_action}")
        else:
            print(f"No action decided this step. LLM Success: {llm_call_success}, Parse Success: {parse_success}")
        print('=' * 50)
        
        return True, final_action
    
    def update(self, action: Optional[str] = None, state: Optional[str] = None):
        """
        update() is called at each time step to update the agent internal state based on the current action and state.
        Memory is could be updated here to keep track of the past actions and states.
        
        Args:
            action: The action string.
            state: The state string.
        
        """
        observation = state
        self.steps += 1
        
        if not isinstance(self.current_sequence, dict):
            print("Error: current_sequence not initialized correctly in update.")
            return
            
        if action is not None and observation is not None:
            # 评估动作结果
            action_failed = self._evaluate_outcome(observation, action)
            self.last_action_successful = not action_failed

            if action_failed:
                self.consecutive_failures += 1
                print(f"INFO: Action '{action}' deemed FAILED. Consecutive failures: {self.consecutive_failures}")
            else:
                self.consecutive_failures = 0
                print(f"INFO: Action '{action}' deemed SUCCESSFUL.")

            # 记录步骤
            step_record = {
                "type": "act_ob_pair",
                "step_id": self.steps-1,
                "action": {
                    "raw_content": action,
                    "llm_time_cost": self.round_llm_duration
                },
                "observation": {
                    "raw_content": observation
                },
                "token_consumption": self.token_counts.copy(),
                "round_time_cost": self.round_cost,
            }
            print(f"Step {self.steps:02} - llm_time_cost: {self.round_llm_duration:.2f}s, round_cost: {self.round_cost:.2f}s")
            self.current_sequence.setdefault("steps", []).append(step_record)
            
            # 更新工具序列日志
            self.tool_sequence.append(action)
            
            # 更新交互历史
            self.interaction_history.append({"action": action, "observation": observation})
            
            # 在记忆中添加最新观察，为下一次LLM调用准备
            if HAS_AutoTool:
                self.memory.add_memory(Message(Role.USER, f"<Observation>{observation}</Observation>\n\n"))
            else:
                self.memory.add_memory(Message(Role.USER, f"<Observation>{observation}</Observation>\n\n"))
                
        elif observation is not None:
            # 这种情况可能发生在只有初始观察更新时
            print(f"Debug: update called with observation only: {observation[:100]}...")
            if HAS_AutoTool:
                self.memory.add_memory(Message(Role.USER, f"<Observation>{observation}</Observation>\n\n"))
            else:
                self.memory.add_memory(Message(Role.USER, f"<Observation>{observation}</Observation>\n\n"))
        else:
            print("Warning: update called without action or observation.")
            
        # 保持记忆大小在限制内
        if len(self.interaction_history) > self.memory_size:
            # 删除最旧的几条交互记录，保留较新的记录
            self.interaction_history = self.interaction_history[-self.memory_size:]
    
    def parse_response_str(self, response_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """解析LLM响应，严格匹配Think:和Action:"""
        if not isinstance(response_str, str): 
            return None, None
            
        response_str = response_str.strip()

        # 更严格的正则表达式，确保Think:和Action:在不同行上
        match = re.search(r"^Think:\s*(.*?)(?:\n|\r\n?)Action:\s*(.+)$", response_str, re.DOTALL | re.MULTILINE)

        if match:
            thought = match.group(1).strip()
            # 获取潜在的动作，剥离额外的行/空格
            action = match.group(2).strip().split('\n')[0].strip()
            if action:  # 确保动作不为空
                return thought, action
            else:
                print(f"Warning: Parsing found 'Action:' but content is empty.")
                return thought, None  # 即使动作为空，也返回思考
        else:
            # 检查是否只有Think:存在，没有后续的Action:
            think_only_match = re.search(r"^Think:\s*(.*?)$", response_str, re.DOTALL | re.MULTILINE)
            if think_only_match:
                thought = think_only_match.group(1).strip()
                # 检查"Action:"是否出现在思考之后的*任何*位置
                if "Action:" not in response_str[think_only_match.end():]:
                    print(f"Warning: Parsing found 'Think:' but no subsequent 'Action:'.")
                    return thought, None
                else:  # Action:存在但格式可能不正确
                    print(f"Warning: Parsing failed. Found 'Think:' and 'Action:' but not in expected sequence/format.")
                    return None, None  # 如果格式错误，表明解析失败
            else:  # 既没有找到Think:也没有找到特定模式
                # 尝试更宽松的匹配，以便更好地诊断问题
                loose_think = re.search(r"Think:\s*(.*?)", response_str, re.DOTALL)
                loose_action = re.search(r"Action:\s*(.+)", response_str, re.DOTALL)
                
                if loose_think and loose_action:
                    print(f"Warning: Loose matching found 'Think:' and 'Action:' but format incorrect.")
                    print(f"Expected format: 'Think:' and 'Action:' on separate lines.")
                    return None, None
                else:
                    print(f"Warning: Parsing failed. Could not find 'Think:...Action:...' structure.")
                    return None, None

    def _print_parsed_response(self, step, thought, action, reflection_influenced=False, is_reflection_step=False):
        """增强版打印函数，显示思考、动作和反思信息"""
        print(f"\nStep {step:02} (Parse OK)")
        
        # 反思影响标记
        reflection_marker = ""
        if is_reflection_step:
            reflection_marker = "🔄 [After Reflection] "
        elif reflection_influenced:
            reflection_marker = "💡 [Reflection Influenced] "
            
        # 打印思考
        print(f"{reflection_marker}Think: {thought}")
        
        # 打印动作
        print('=' * 15, "  Action  ", '=' * 15) 
        print(f"Action: {action}")
        
    def _validate_reflection_format(self, reflection):
        """验证反思格式是否符合标准"""
        if not reflection or not isinstance(reflection, str):
            return False, "Empty or invalid reflection type"
            
        # 检查是否以"Self-Reflection:"开头
        if not reflection.strip().startswith("Self-Reflection:"):
            return False, "Missing 'Self-Reflection:' prefix"
            
        content = reflection.replace("Self-Reflection:", "").strip()
        if len(content) < 50: # Increased minimum length for more detailed reflection
            return False, "Reflection content too short (expected more detail)"
            
        # 检查是否包含指示分析和计划的关键词
        analysis_keywords = ["reason for failure", "flaw in reasoning", "error in", "mistake was", "should have"]
        plan_keywords = ["corrected plan", "new plan", "next steps", "try to", "instead i will"]
        
        has_analysis = any(kw in content.lower() for kw in analysis_keywords)
        has_plan = any(kw in content.lower() for kw in plan_keywords)
        
        if not (has_analysis and has_plan):
            missing_parts = []
            if not has_analysis: missing_parts.append("failure analysis")
            if not has_plan: missing_parts.append("corrected plan")
            return False, f"Missing key components: {', '.join(missing_parts)}"
            
        return True, "Valid reflection format"

    def _generate_reflection(self) -> str:
        """根据任务类型生成反思，分析过去的行为并提出改进计划"""
        print("生成反思...")
        
        # 准备反思提示
        scenario = self._get_scenario_from_history()
        last_action = self.last_action
        # repeat_count = self.consecutive_same_action_count # No longer the primary trigger
        failure_context = f"The agent experienced {self.consecutive_failures} consecutive failures, with the last action being '{last_action}'."

        reflection_prompt_template = f"""
You are an AI assistant analyzing a problematic interaction in a {{task_type}} environment.
The agent is struggling to achieve the goal: {{goal}}
{failure_context}

Recent Interaction History (leading to failure):
{{scenario}}

Task: Analyze the situation and provide a detailed self-reflection that can guide future actions. This reflection should act as a 'semantic gradient'.
Your reflection MUST:
1. Identify the likely REASON for the recent failure(s). What went wrong with the agent's approach or understanding?
2. Explain the FLAW in the agent's reasoning or strategy given the observations and goal.
3. Propose a CORRECTED, step-by-step plan of action. This plan should be specific and aim to overcome the identified error and make progress towards the goal.
4. Be concise yet comprehensive.

Format your response starting EXACTLY with "Self-Reflection:".
Example:
Self-Reflection:
Reason for failure: I repeatedly tried to pick up an object that was not visible or accessible from my current location.
Flaw in reasoning: I assumed the object was present without confirming its location or accessibility through 'look' or 'go to' actions first.
Corrected plan:
1. Use 'look' to carefully survey the current area and identify key objects and receptacles.
2. If the target object or a necessary intermediate object/receptacle is not visible, use 'go to [relevant_location]' to navigate to a more appropriate area based on the goal.
3. Once in the correct location and the object is visible, then attempt the 'take [object] from [receptacle]' action.
"""

        # 根据任务类型选择不同的反思提示模板
        # (Placeholders for more specific prompts per task if needed, current template is fairly general)
        if self.task_type.startswith("alfworld"):
            task_specific_guidance = "Consider common Alfworld mistakes like incorrect object states, missing prerequisites for actions (e.g., opening containers), or navigation errors."
            reflection_prompt = reflection_prompt_template.format(
                task_type="ALFWorld",
                goal=self.goal,
                scenario=scenario,
                task_specific_guidance=task_specific_guidance
            )
        elif self.task_type.startswith("scienceworld"):
            task_specific_guidance = "Think about scientific principles, properties of substances, correct use of tools (e.g., thermometer, scale), and necessary experimental procedures."
            reflection_prompt = reflection_prompt_template.format(
                task_type="ScienceWorld",
                goal=self.goal,
                scenario=scenario,
                task_specific_guidance=task_specific_guidance
            )
        elif any(self.task_type.startswith(t) for t in ["tool_query", "weather", "movie"]):
            task_specific_guidance = "Focus on correct API call sequences, parameter formatting, and understanding tool capabilities. Did the agent use the right tool for the sub-task?"
            reflection_prompt = reflection_prompt_template.format(
                task_type="Tool-Based Query",
                goal=self.goal,
                scenario=scenario,
                task_specific_guidance=task_specific_guidance
            )
        else:
            # 默认反思提示
            task_specific_guidance = "Consider general problem-solving strategies: breaking down the problem, checking assumptions, and exploring alternative actions."
            reflection_prompt = reflection_prompt_template.format(
                task_type=self.task_type,
                goal=self.goal,
                scenario=scenario,
                task_specific_guidance=task_specific_guidance
            )
        
        try:
            # 使用独立的API调用生成反思，以确保不影响主记忆
            if HAS_AutoTool:
                messages = [
                    {"role": "system", "content": f"You are an expert analytical assistant that helps an agent learn from its mistakes in a {self.task_type} environment by providing structured self-reflections."},
                    {"role": "user", "content": reflection_prompt}
                ]
                print(f'[Evaluator DEBUG]: llm_input_messages: {messages}')
                reflection, incount_token, outcount_token = call_model(messages)
                self.current_sequence["metadata"]["total_input_token"] += incount_token
                self.current_sequence["metadata"]["total_output_token"] += outcount_token
                self._increment_llm_calls()

            else:
                print('Could not generate reflection: AutoTool call_model not available')
                return f"Self-Reflection:\nReason for failure: The agent was stuck repeating '{last_action}'.\nFlaw in reasoning: Lack of adaptation after repeated failures.\nCorrected plan: Analyze the environment carefully. Try a different sequence of actions. Check inventory and available actions if unsure."
            
            # 格式化反思，确保以"Self-Reflection:"开头
            if reflection and not reflection.strip().startswith("Self-Reflection:"):
                reflection = "Self-Reflection:\n" + reflection.strip()
            elif not reflection:
                 return f"Self-Reflection:\nReason for failure: LLM failed to provide reflection content for action '{last_action}'.\nFlaw in reasoning: Undetermined due to LLM error.\nCorrected plan: Attempt to re-evaluate the situation and try a different approach. If stuck, use 'look' or 'check valid actions'."

                
            # 验证反思格式 (轻量级检查，因为LLM可能不完全遵循复杂格式)
            is_valid, message = self._validate_reflection_format(reflection)
            if not is_valid:
                print(f"⚠️ Warning: Generated reflection has issues - {message}. Attempting to use anyway.")
                # 即使格式不完美，也尝试使用，但确保前缀正确
                if not reflection.strip().startswith("Self-Reflection:"):
                    reflection = "Self-Reflection:\n" + reflection.strip()
                
            print(f"Generated Reflection: {reflection}")
            return reflection.strip()
            
        except Exception as e:
            print(f"Reflection generation failed: {e}")
            return f"Self-Reflection:\nReason for failure: An exception occurred during reflection generation for action '{last_action}'. Error: {e}.\nFlaw in reasoning: Undetermined due to exception.\nCorrected plan: Recover from the error. Re-evaluate the current state using 'look' and then try a different action or strategy."
            
    def get_reflection_stats(self):
        """返回反思统计信息，用于调试和分析"""
        return {
            "total_reflections": self.reflection_count,
            "reflection_timestamps": self.reflection_timestamps,
            "reflection_threshold": self.reflection_threshold,
            "reflection_enabled": self.enable_reflection,
            "current_consecutive_actions": self.consecutive_same_action_count,
            "reflections": self.reflections
        }
        
    def print_reflection_history(self):
        """打印所有历史反思"""
        if not self.reflections:
            print("No reflections generated yet.")
            return
            
        print("\n" + "="*50)
        print(f"REFLECTION HISTORY ({self.reflection_count} total)")
        print("="*50)
        
        for i, (reflection, step) in enumerate(zip(self.reflections, self.reflection_timestamps)):
            print(f"\nReflection #{i+1} (Step {step}):")
            print("-"*40)
            print(reflection)
            print("-"*40)
        print("\n")
        
    def _get_scenario_from_history(self) -> str:
        """从交互历史中提取相关场景信息"""
        scenario = f"Goal: {self.goal}\nInitial observation: {self.init_obs}\n\nInteractions:\n"
        
        # 添加最后N个交互以创建场景
        for interaction in self.interaction_history[-10:]:
            scenario += f"> {interaction['action']}\n{interaction['observation']}\n"
            
        return scenario
    
    def _update_system_prompt(self):
        """更新系统提示以包含新的反思"""
        system_prompt = self._make_sys_prompt()
#         print(f'[DEBUG INFO]: System prompt content: {system_prompt}')
        # 用新的系统提示替换旧的
        if HAS_AutoTool:
            memory_messages = self.memory.get_memory(parse=True)
            if memory_messages and memory_messages[0]['role'] == 'system':
                # 创建新的记忆实例
                new_memory = TemporaryMemory()
                # 添加新的系统提示
                new_memory.add_memory(Message(Role.SYSTEM, system_prompt))
                # 添加其余消息（排除原始系统消息）
                for msg in memory_messages[1:]:
                    new_memory.add_memory(Message(msg['role'], msg['content']))
                self.memory = new_memory
        else:
            # 简单的备用实现
            messages = self.memory.get_memory(parse=True)
            if messages and messages[0]['role'] == 'system':
                messages[0]['content'] = system_prompt
    
    def _increment_llm_calls(self):
        """递增当前轨迹的LLM调用计数"""
        if isinstance(self.current_sequence, dict) and "metadata" in self.current_sequence:
            self.current_sequence["metadata"]["total_llm_calls"] += 1
            
    def _finalize_and_log_trajectory(self, status="Completed", summary="Trajectory finished."):
        """完成当前轨迹日志并将其附加到主日志中"""
        if not isinstance(self.current_sequence, dict):
            return

        end_time = datetime.now()
        start_time_str = self.current_sequence["metadata"]["start_time"]
        try:
            start_time = datetime.strptime(start_time_str, "%Y%m%d_%H%M%S.%f")
            total_time_seconds = (end_time - start_time).total_seconds()
        except ValueError:
            start_time = None
            total_time_seconds = self.cumulative_time

        # 更新元数据
        self.current_sequence["metadata"]["end_time"] = end_time.strftime("%Y%m%d_%H%M%S.%f")
        self.current_sequence["metadata"]["total_time_seconds"] = round(total_time_seconds, 3)
        self.current_sequence["metadata"]["status"] = status

        # 添加累计统计信息到轨迹日志
        self.current_sequence['cumulative_llm_time'] = round(self.cumulative_time, 3)

        self.current_sequence["summary"] = summary

        # 将完成的序列附加到主日志列表
        self.action_log.setdefault("sequences", []).append(self.current_sequence)
        self.action_log["total_sequences"] = len(self.action_log["sequences"])
        self.action_log["processed_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 保存整个日志文件
        self._save_action_log_file()

        # 清除current_sequence以供下一次运行
        self.current_sequence = None

    def _save_action_log_file(self):
        """将整个action_log字典保存到JSON文件"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.action_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving log file {self.log_file}: {e}")

    # --- 辅助方法 ---
    def get_action_sequence(self) -> List[str]:
        """返回当前轨迹中执行的工具名称序列"""
        return self.tool_sequence

    def get_current_trajectory_log(self) -> Optional[Dict[str, Any]]:
         """返回当前活动轨迹的日志数据"""
         return self.current_sequence

    def finalize_and_save(self, status="Manually Finalized", summary="Logging explicitly finalized."):
         """手动完成当前轨迹并保存日志文件"""
         print("Finalizing and saving logs...")
         self._finalize_and_log_trajectory(status=status, summary=summary)
         print(f"Log saved to {self.log_file}")

    @classmethod
    def from_config(cls, llm_model, config):
        """从配置创建代理"""
        init_prompt_path = config.get("init_prompt_path", None)
        
        return cls(
            llm_model=llm_model,
            task_type=config.get("task_type", "tool-query"),
            memory_size=config.get("memory_size", 100),
            examples=config.get("examples", []),
            instruction=config.get("instruction", ""),
            init_prompt_path=config.get("init_prompt_path", '/data/agentboard/prompts/ReactBaselineAgent/academic_prompt.json'),
            system_message=config.get("system_message", "You are a helpful assistant."),
            check_actions=config.get("check_actions", "check valid actions"),
            check_inventory=config.get("check_inventory", "inventory"),
            max_think_iters=config.get("max_think_iters", 3),
            reflection_threshold=config.get("reflection_threshold", 3),
            few_shot_examples_path=config.get("few_shot_examples_path", "./alfworld_runs/reflexion_few_shot_examples.txt"),
            prompts_json_path=config.get("prompts_json_path", "./alfworld_runs/prompts/alfworld_3prompts.json"),
            debug=config.get("debug", False),
            record_mod=config.get("record_mod", True),
            tag=config.get("tag", "reflection_agent"),
            enable_reflection=config.get("enable_reflection", True)
        )