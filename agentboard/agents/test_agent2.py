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

# 方式1：自定义 JSON 编码器
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)
    
# --- 惯性组件导入 ---
try:
    from autool.core.tool_predict.datastruct import ToolGraph
# #     # print('[DEBUG INFO]: ToolGraph is inited when imported?')
    # 使用包含 record_action 的 ALFWorld 版本
    from autool.core.param_completion.domain.alfworld import AlfworldParamCompletion 
    from autool.core.param_completion.domain.tool_query import ToolQueryParamCompletion
    from autool.core.param_completion.domain.scienceworld import ScienceWorldParamCompletion
    from agentboard.prompts.ReflectionBaselineAgent.academic_sys_prompt import TOOL_QUERY_REFLECTION_EXAMPLES, TOOL_QUERY_REFLECTION_PROMPT
    from agentboard.prompts.ReflectionBaselineAgent.alfworld_sys_prompt import ALFWORLD_REFLECTION_EXAMPLES, ALFWORLD_REFLECTION_PROMPT
    from agentboard.prompts.ReflectionBaselineAgent.scienceworld_sys_prompt import SCIENCEWORLD_REFLECTION_EXAMPLES, SCIENCEWORLD_REFLECTION_PROMPT
    from agentboard.prompts.ReflectionBaselineAgent.reflection_instruction import REFLECTION_INSTRUCTIONS
    # *******************************************************
    HAS_INERTIA_COMPONENTS = True
except ImportError as e:
    print(f"警告: 未找到惯性组件 (in my agent): {e}")
    HAS_INERTIA_COMPONENTS = False
    

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

SCIENCEWORLD_REFLECTION_PROMPT = """
You are an intelligent scientist.
Your MAIN GOAL is: **{goal}**
You MUST achieve ALL aspects of this goal. Do not simplify or ignore parts of the goal.

Follow the ReAct (Reasoning + Acting) paradigm:
1.  **Think:**
    a.  **Analyze Observation:** What are the key objects, substances, locations, and their states in the current observation? What new information is available? Pay attention to details like temperature, contents of containers, and device states.
    b.  **Assess Goal Progress:** How does the current situation relate to the MAIN GOAL ({goal})? If the goal involves changes of state, specific temperatures, or combinations of substances, which parts are met and which are pending? What is the next logical step towards FULL goal completion?
    c.  **Plan Action:** Detail your reasoning for the *specific* action you will take. If previous attempts to achieve a sub-goal failed (e.g., an object didn't change state as expected, a device didn't activate), explain your new strategy. Consider the properties of substances and the functions of devices.
    d.  **Analyze Failure (If Applicable):** If the previous action resulted in an unexpected or undesired outcome, analyze what went wrong and adjust your approach.
2.  **Action:** Output EXACTLY ONE command from the list below based on your thought process.

Available Actions:
- **Manipulation**:
  - open {{OBJ}} / close {{OBJ}}: Interact with a container.
  - pick up {{OBJ}}: Add an object to your inventory.
  - put down {{OBJ}}: Remove an object from your inventory.
  - move {{OBJ}} to {{OBJ}}: Transfer an object.
  - pour {{OBJ}} into {{OBJ}}: Pour a substance.
  - dunk {{OBJ}} into {{OBJ}}: Immerse a container in a liquid.
  - mix {{OBJ}}: Chemically combine contents within a container object.

- **Inspection**:
  - look around: Survey your surroundings.
  - look at {{OBJ}}: Examine an object or substance closely.
  - look in {{OBJ}}: Peek inside a container.
  - read {{OBJ}}: Review written content on an object.

- **Device Operations**:
  - activate {{OBJ}} / deactivate {{OBJ}}: Toggle a device (e.g., stove, sink, lighter).
  - use {{OBJ}} [on {{OBJ}}]: Utilize a device or item. If 'on {{OBJ}}' is used, it specifies a target for the first OBJ.

- **Movement**:
  - go to {{LOC}}: Relocate to a new location (e.g., go to kitchen).

- **Miscellaneous**:
  - eat {{OBJ}}: Consume an edible item.
  - flush {{OBJ}}: Activate a flushing mechanism (e.g., toilet).
  - focus on {{OBJ}}: Direct attention to a particular object or substance.
  - wait or wait [DURATION]: Pause for a specified period.

- **Information**:
  - task: Recap your current objective.
  - inventory: Display items you're carrying.

Where:
- {{OBJ}}: Represents an object or substance name (e.g., 'metal pot', 'water', 'thermometer').
- {{LOC}}: Represents a location name (e.g., 'kitchen', 'hallway').
- [DURATION]: Represents a number for wait time (e.g., '1', '5').

Guidelines for Action:
*   Use object, substance, and location names *EXACTLY* as they appear in the LATEST observation.
*   Refer to the interaction examples below for typical patterns, but adapt to the current goal and observation.
*   For actions requiring parameters like {{OBJ}} or {{LOC}}, replace them with specific names from the observation.
*   Pay attention to the current focused object/substance. Some actions implicitly use the focused item.
*   **Task Completion:** The task is considered complete by the environment ONLY when ALL conditions of the MAIN GOAL are met. There is no explicit 'finish' command. If you believe all conditions are met, your final action should be the one that directly achieves the last part of the goal. The environment will then typically provide a success message or end the episode. If not, re-assess if all goal conditions were truly satisfied.

**CRITICAL FORMATTING:** Your response MUST be structured EXACTLY as follows, with "Think:" and "Action:" on SEPARATE lines and NO extra text before or after:
Think: [Your reasoning and plan here, addressing points a-d under 'Think:']
Action: [Your single, valid ScienceWorld command here from 'Available Actions']


**Interaction Examples:**
---
{examples_str}
---


{reflection_instructions}
"""

# from agentboard.prompts.Reflection.reflection_instructions import get_reflection_instructions

# --- AutoTool / Memory 导入 (带回退逻辑) ---
from autool.utils import call_model
from autool.message import Message, Role
from autool.memory import TemporaryMemory
from autool.utils.parser.alfworld import parse_alfworld_action
from autool.utils.parser.tool_query import parse_tool_query
from autool.utils.parser.scienceworld import parse_scienceworld_action
from autool.utils.parser.check_action import check_tool_failure
HAS_AutoTool=True

    

@registry.register_agent("ReflectionInertialAgent")
class ReflectionInertialAgent(BaseAgent):
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
                use_parser=True,
                few_shot_examples_path="./alfworld_runs/reflexion_few_shot_examples.txt",
                prompts_json_path="./alfworld_runs/prompts/alfworld_3prompts.json",
                debug=False,
                record_mod=True,
                tag="reflection_agent",
                enable_reflection=True,  # 是否启用反思功能
                # --- 新增: 惯性配置 ---
                use_inertia=True, # 惯性功能的总开关
                inertia_threshold=0.2, # 触发参数填充的置信度阈值
                inertia_alpha=0.5, # 频率与语义得分的权重
                inertia_k=2, # 用于预测的历史窗口大小
                inertia_max=1, # 最长的连续惯性调用链
                inertia_fallback_hint=True, # 惯性失败时是否给 LLM 提示?
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
        self.use_parser = use_parser
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
        # --- 日志与历史 ---
        # self.tool_sequence = [] # 历史现在由 param_completion 管理
        log_dir = os.path.join(os.getcwd(), 'logs'); os.makedirs(log_dir, exist_ok=True)
        # 使用不同的日志文件名以区别于基线
        self.log_file = os.path.join(f'/data/agentboard/examples/{self.task_type}_{self.tag}/trajectories', f"{self.task_type}_inertia_agent_calls_{time.strftime('%Y%m%d_%H%M%S')}.json")
        self.action_log = {
            "start_time": time.time(),
            "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_sequences": 0, "total_tool_calls": 0, "total_inertial_calling": 0, "sequences": [], "overhead": {} }
        self.current_sequence = {}
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
        

        # --- 惯性组件导入 ---
        self.use_inertia = use_inertia and HAS_INERTIA_COMPONENTS
        self.inertia_threshold = inertia_threshold
        self.inertia_alpha = inertia_alpha
        self.inertia_k = inertia_k
        self.inertia_window = deque(maxlen=inertia_k+1) # 用于存储最近的 k 次调用
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
                print("正在初始化惯性组件...")
                if not os.path.exists(self.log_file):
                    print(f"创建日志文件目录: {os.path.dirname(self.log_file)}")
                    os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                # action_description = '/data/agentboard/examples/tool_description/alfworld_action_description.json'
# #                 # print(f'[DEBUG INFO]: action_description is: {action_description}, log_file is: {self.log_file}')
                # tool_description = '/data/agentboard/prompts/ReactInertiaAgent/tool_description.json'
                # 创建日志文件
                if not os.path.exists(self.log_file):
                    print(f"创建日志文件: {self.log_file}")
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
# #                 print(f'[DEBUG INFO]: tool_description_path is: {tool_description_path}')
# #                 print(f'[DEBUG INFO]: log_file is: {self.log_file}')
                # self.tool_graph.load_from_json(tool_description_path, '/data/logs/scienceworld_react_baseline_log_20250512_144931.json')
                self.tool_graph.load_from_json(tool_description_path, self.log_file)
                # --- 检查是否加载成功工具图 ---
# #                 print('f[DEBUG INFO]: check if tool graph is loaded successfully')
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
            print("惯性组件不可用或已通过配置禁用。")
            self.tool_graph = None
            self.param_completion = None
            self.param_dependency_path = None
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
    def _init_param_completion(self, tool_description=None):
        try:

            if self.task_type == 'alfworld':    
                self.param_completion = AlfworldParamCompletion(self.tool_graph,
                                                                dependency_graph_path=self.log_file,
                                                                tool_descriptions=tool_description,
                                                                history_max_len=20,
                                                                goal=self.goal,
                                                                debug=self.debug
                )
                # self.param_completion = ALFWorldParamCompletion(self.tool_graph)
            elif self.task_type in ['tool_query', 'tool-query', 'tool-operation', 'academic']:
                print(f'[debug] start to init ToolQueryParamCompletion')
                self.param_completion = ToolQueryParamCompletion(self.tool_graph,
                                                                    dependency_graph_path=self.log_file,
                                                                    tool_descriptions=tool_description,
                                                                    history_max_len=20,
                                                                    goal=self.goal,
                                                                    debug=self.debug)
# #                 print('f[DEBUG INFO]: ToolQueryParamCompletion is inited')
            elif self.task_type == 'scienceworld':
# #                 print(f'[DEBUG INFO]: ScienceWorldParamCompletion is inited')
                self.param_completion = ScienceWorldParamCompletion(self.tool_graph,
                                                                    dependency_graph_path=self.log_file,
                                                                    tool_descriptions=tool_description,
                                                                    history_max_len=20,
                                                                    goal=self.goal,
                                                                    debug=self.debug)
        except Exception as e:
            print(f"错误：初始化参数填充组件失败: {e}。")
            self.param_completion = None

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
# #         print(f'[DEBUG INFO]: task_type: {self.task_type}')
        if self.task_type == 'alfworld':
            system_template = self.sys_prompts["alfworld"]
        elif self.task_type == "scienceworld":
            system_template = self.sys_prompts["scienceworld"]
        elif self.task_type in ['tool_query', 'tool-query', 'tool-operation', 'academic']:
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
            
                
            # 格式化模板
            system_prompt_content = system_template.format(**format_args)
            return system_prompt_content
            
        except Exception as e:
            print(f"Error formatting system prompt: {e}")
            # 出错时回退到简单格式
            return f"Your goal is: {self.goal}\n\n{reflection_instructions}"


    def _parse_raw_action(self, action: str) -> Optional[Dict[str, Any]]:
        """
        最后返回的统一格式
        parsed_action = {"tool_name": "unknown", "inputs": {}}
        """
        parsed_action_result = None
        if self.task_type == "alfworld":
            print(f"Debug: Parsing action: {action}")
            parsed_action_result = parse_alfworld_action(action)
        elif self.task_type in ["tool_query", "tool-query", "academic"]: 
            parsed_action_result = parse_tool_query(action)
            print(f"Parsed Action: {parsed_action_result}")
        elif self.task_type == "scienceworld":
            parsed_action_result = parse_scienceworld_action(action)
            print(f"Parsed Action: {parsed_action_result}")        
        return parsed_action_result
    

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
        self.param_completion.framework.history.history = []
        self.goal = goal
        self.init_obs = init_obs
        self.steps = 0
        self.think_count = 0
        self.tool_sequence = []
        self.cumulative_time = 0.0
        self.inertia_count = 0
        self.reflections = []
        self.interaction_history = []
        self.consecutive_same_action_count = 0
        self.last_action = ""
        self.init_obs = init_obs
        # 确定任务前缀，用于选择合适的示例
        self.task_prefix = self._determine_task_prefix(goal)
        print(f"Task prefix determined: {self.task_prefix}")
        # --- 重置惯性状态 ---
        if self.use_inertia and self.param_completion:
            print("正在重置惯性状态...")
            try:
                self.param_completion.adapter.reset()
                # self.param_completion.adapter.update_state(init_obs)
            except Exception as e:
                print(f"重置时更新场景知识出错: {e}")
            print("惯性状态已重置。")


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
            "inertial_step": [], # Optional: For environment rewards
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
# #         # print(f'[DEBUG INFO]: System prompt content: {system_prompt_content}')
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
        self.round_llm_duration = 0.0 # 每轮的 LLM 调用时间
        
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

        ##########################################################
        ####                加入惯性优化框架                     ####
        #### modified from test_agent.py by jjy on 5.13.2025  ####
        ###########################################################
        # # --- inertial param ---
        execute_inertially = False # 标记是否最终通过惯性执行跳过了 LLM
        inertia_prediction_made = False # 标记是否进行了预测尝试
        filling_successful = False # 标记参数填充是否成功 (仅当尝试惯性执行时)

        inertial_flag = True # 标记当前轮次是否考虑惯性

        # # --- 1. 尝试惯性预测和填充 (在 LLM 循环之前) ---
        if self.continuous_inertial_call_count >= self.inertia_max or self.inertia_count / (self.steps + 1) > 0.3:
            inertial_flag = False
            self.continuous_inertial_call_count = 0

        # 

        if self.current_sequence["steps"][-1]['type'] != "initial":
            print(self.current_sequence["steps"][-1])
            if self.current_sequence["steps"][-1]["action"]["llm_time_cost"] == 0.0:
                print("成功阻止连续惯性调用")
                inertial_flag = False
                self.continuous_inertial_call_count = 0 

        
        # --- 检查是否启用并满足惯性预测条件 ---
        can_predict = (inertial_flag and
                       self.use_inertia and
                       self.param_completion and
                       self.tool_graph and
                       len(self.param_completion.framework.history.history) > self.inertia_k) # 使用 ExecutionHistory

# #         print(f'[DEBUG INFO]: self.param_completion.framework.history.history: {len(self.param_completion.framework.history.history)}')
# #         # print(f'[DEBUG INFO]: inertia_count_sum: {self.inertia_count_sum}, steps: {self.steps}')
        if can_predict:
            # 使用 param_completion 中的 history (结构化记录)
            # 注意：确保 history 存储的是 tool_name 列表或类似结构
            # 如果 history 存储的是完整字典，需要先提取 tool_name
            # 假设 param_completion 有一个方法或属性可以获取工具名历史
            # current_history_types = self.param_completion.get_tool_name_history(last_k=self.inertia_k)
            # 简化：直接从 ExecutionHistory 获取 tool_name
            recent_records = self.param_completion.framework.history.get_latest_records(k=self.inertia_k)
            recent_history_types = [rec.get("tool_name") for rec in recent_records if rec.get("tool_name")]

            if recent_history_types: # 需要有有效的历史工具名
                if self.debug: print(f"--- 惯性检查: 当前历史 (最近 {min(self.inertia_k, len(recent_history_types))} 步): {recent_history_types} ---")

                current_thought = self.memory.get_memory(parse=True)[-2]["content"]
                intuition = f"{current_thought}"
                try:
                    s_time = time.time()
                    has_inertia, inertial_sense_overhead, predicted_type = self.tool_graph.predict_next_tool_with_chain_similarity(
                        recent_history_types,
                        intuition=intuition,
                        thereshold=self.inertia_threshold, # 在这里使用阈值判断是否高置信度
                        alpha=self.inertia_alpha
                    )
                    self.total_inertial_sense_overhead = time.time() - s_time
                    # if "go to" in predicted_type:
                    #     has_inertia = False
                    # if "finish" in predicted_type:
                    #     has_inertia = False
                    inertia_prediction_made = True # 标记已进行预测尝试
                    self.total_graph_search_time += inertial_sense_overhead["search_time"]
                    self.total_SimSCE_time += inertial_sense_overhead["SimSCE_time"]
                    self.total_intuition_embedding_time += inertial_sense_overhead["intuition_embedding_time"]
                    self.total_path_embedding_time += inertial_sense_overhead["path_embedding_time"]
                    self.total_similarity_cost_time += inertial_sense_overhead["similarity_cost_time"]

                    if has_inertia and predicted_type: # 如果预测结果置信度高
                        print(f"--- 惯性预测: 高置信度预测为 '{predicted_type}' (得分 > {self.inertia_threshold}) ---")
                        print(f"--- 尝试为 '{predicted_type}' 进行参数填充 ---")
                        # 尝试参数填充
                        s_time = time.time()
                        self.param_filling_meta_data, filled_params = self.param_completion.fill_parameters(
                            predicted_type, {}, self.inertia_k
                        )
# #                         if self.debug: print("[DEBUG INFO]: filled_params is: ", filled_params)
                        # 检查填充是否成功
                        filling_successful = self._check_param_filling_success(predicted_type, filled_params)
                        self.total_param_filling_time += time.time() - s_time

                        if filling_successful:
                            print(f"--- 惯性成功: '{predicted_type}' 的参数已填充 ---")
                            # 生成动作字符串
                            s_time = time.time()
                            final_action = self.param_completion.adapter.generate_action_from_params(predicted_type, filled_params)
                            self.total_generate_action_time += time.time() - s_time
                            print(f"--- 执行惯性动作: {final_action} ---")
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
                    self.total_llm_time += self.round_llm_duration
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
        parsed_action_result = None
        parsed_observation_result = None
        # --- 解析动作 ---
        if action and self.use_parser:
            s_time = time.time()
            parsed_action_result = self._parse_raw_action(action)
            self.total_parser_time += time.time() - s_time
        # --- 解析 observation ---
        if observation and self.use_parser:
            s_time = time.time()
            parsed_observation_result = self.param_completion.adapter.infer_output(parsed_action_result["tool_name"], parsed_action_result["inputs"], observation)
            self.total_parser_time += time.time() - s_time

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

# #             print(f'[DEBUG INFO] parsed_observation_result {parsed_observation_result}')
            # 记录步骤
            step_record = {
                "type": "act_ob_pair",
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

        # --- 对于惯性调用加入反馈机制 ---
        self.inertia_window.append(parsed_action_result["tool_name"])
        list_inertial_window = list(self.inertia_window)
        list_inertial_window = list(self.inertia_window)

        # 找出列表中出现次数最多的元素及其出现次数
        most_common_element = max(set(list_inertial_window), key=list_inertial_window.count)
        count_most_common = list_inertial_window.count(most_common_element)
        if self.round_llm_duration == 0.0:
            if check_tool_failure(observation, self.task_type):
                print(f"Warning: 发现工具惯性调用失败，加入负权边 {list_inertial_window}")
                self.tool_graph.record_tool_sequence(list_inertial_window, weight=-1)
            else:
                print(f"Warning: 发现工具惯性调用成功，加入正权边 {list_inertial_window}")
                self.tool_graph.record_tool_sequence(list_inertial_window, weight=1)
            
        else:
            if check_tool_failure(observation, self.task_type):
                print(f"Warning: 发现工具LLM调用失败，加入负权边 {list_inertial_window}")
                self.tool_graph.record_tool_sequence(list_inertial_window, weight=-0.5)
        # --- 新增: 使用 ParamCompletion 记录动作 ---
        # 必须在每一步之后调用以维护历史记录
        if self.use_inertia and self.param_completion and action and state:
            try:
                # 这个方法会根据动作和观察更新 param_completion 内部的
                # executed_tools, tool_outputs, inventory, objects 等。
                self.param_completion.framework.record_execution(action, parsed_action_result, state)
                if self.debug: print(f"为惯性跟踪记录的动作: {action}")
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
                # print(f'[Evaluator DEBUG]: llm_input_messages: {messages}')
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
# #         # print(f'[DEBUG INFO]: System prompt content: {system_prompt}')
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
        self.action_log["inertial_step"] = self.inertial_step
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
        
        # 保存整个日志文件
        # print(f"Saving action log to {self.action_log}...")
        self._save_action_log()

        print('*'*100)
# #         # print(f'[DEBUG INFO] self.current_sequence: {self.current_sequence}')
        print('*'*100)
        # --- 保存当前轨迹 ---
        s_time = time.time()
        # --- 更新 ToolGraph ---
        if self.use_inertia and self.tool_graph:
            try:
# #                 print(f'[DEBUG INFO] self.current_sequence: {self.current_sequence}')
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
        """将整个action_log字典保存到JSON文件"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.action_log, f, cls=SetEncoder, indent=2, ensure_ascii=False)
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

    def _check_param_filling_success(self, predicted_next_type: str, filled_params: Dict[str, Any]) -> bool:
        """辅助函数，检查是否所有必需的参数都已填充。"""
        # 特判
# #         print(f"[DEBUG INFO]: predicted_next_type is: {predicted_next_type}, filled_params is: {filled_params}")
        if not self.use_inertia or not self.tool_graph: return False
        if predicted_next_type not in self.tool_graph.nodes: return False # 无法检查需求

        tool_node = self.tool_graph.nodes[predicted_next_type]
        required_param_names = set(tool_node.input_params.keys())
# #         print("[DEBUG INFO]: required_param_names is: ", required_param_names)
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
        
    @classmethod
    def from_config(cls, llm_model, config):
        """从配置创建代理"""
        init_prompt_path = config.get("init_prompt_path", None)
        
        return cls(
            llm_model=llm_model,
            task_type=config.get("task_type", "scienceworld"),
            memory_size=config.get("memory_size", 100),
            examples=config.get("examples", []),
            instruction=config.get("instruction", ""),
            init_prompt_path=config.get("init_prompt_path", '/data/agentboard/prompts/ReactBaselineAgent/academic_prompt.json'),
            system_message=config.get("system_message", "You are a helpful assistant."),
            check_actions=config.get("check_actions", "check valid actions"),
            check_inventory=config.get("check_inventory", "inventory"),
            max_think_iters=config.get("max_think_iters", 3),
            use_parser=config.get("use_parser", True),
            reflection_threshold=config.get("reflection_threshold", 3),
            few_shot_examples_path=config.get("few_shot_examples_path", "./alfworld_runs/reflexion_few_shot_examples.txt"),
            prompts_json_path=config.get("prompts_json_path", "./alfworld_runs/prompts/alfworld_3prompts.json"),
            debug=config.get("debug", False),
            record_mod=config.get("record_mod", True),
            tag=config.get("tag", "bare惯性+reflection"),
            enable_reflection=config.get("enable_reflection", True),
            # --- 传递惯性配置 ---
            use_inertia=config.get("use_inertia", True),
            inertia_threshold=config.get("inertia_threshold", 0.1),# 0.15
            inertia_alpha=config.get("inertia_alpha", 0.5),
            inertia_k=config.get("inertia_k", 2),
            inertia_max=config.get("inertia_max", 1),
            inertia_fallback_hint=config.get("inertia_fallback_hint", True),
        )