import json
import time
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from common.registry import registry
from agents.base_agent import BaseAgent
from agentboard.prompts.ReactBaselineAgent.alfworld_sys_prompt import ALFWORLD_SYS_PROMPT
from agentboard.prompts.ReactBaselineAgent.tool_query_sys_prompt import TOOL_QUERY_SYSTEM_PROMPT
from agentboard.prompts.ReactBaselineAgent.scienceworld_sys_prompt import SCIENCEWORLD_SYS_PROMPT


# --- AutoTool / Memory Imports (with Fallback) ---
try:
    from autool.message import Message, Role
    from autool.memory import TemporaryMemory # Using this as the primary memory
    from autool.utils.parser.alfworld import parse_alfworld_action
    from autool.utils.parser.tool_query import parse_tool_query
    from autool.utils.parser.scienceworld import parse_scienceworld_action
    from autool.utils import call_model
    # 导入Toolkit
    from autool.tools.toolkit import Toolkit, get_tool_desc
    HAS_AutoTool = True
except ImportError:
    HAS_AutoTool = False

# --- Logging Setup ---
log_dir = os.path.join(os.getcwd(), "logs")
if not os.path.exists(log_dir): os.makedirs(log_dir)
# Optional: Setup file logging if needed
log_file = os.path.join(log_dir, f"/console/agent_log_{time.strftime('%Y%m%d_%H%M%S')}.log")
# logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')


@registry.register_agent("ReactBaselineAgent")
class ReactBaselineAgent(BaseAgent):
    """
    A baseline agent implementing the ReAct paradigm for Alfworld-like environments.
    Includes improved logging with action-observation pairing and placeholders for action parsing.
    """
    def __init__(self,
                llm_model, # The LLM model instance/client 
                task_type="", # Task type (for future use)
                memory_size=100, # Max history turns (approx)
                examples=[],
                instruction="", # General instruction (less critical now)
                init_prompt_path=None,
                system_message="You are a helpful assistant.", # Base identity
                need_goal=True, # Should always be True for this agent type
                check_actions="check valid actions", # Help command
                check_inventory="inventory", # Help command
                use_parser=True, # Use the action parser?
                max_think_iters=3, # Max consecutive think steps without action
                max_steps=50, # Max env steps (used by external loop usually)
                debug=False, # Set to True for verbose logs
                record_mod=True, # Record run stats for each step 
                toolkit=None, # Added toolkit parameter
                ):
        super().__init__()
        # --- Core Attributes ---
        self.llm_model = llm_model # NOTE: Currently only used for potential token counting
        if not HAS_AutoTool and llm_model:
            print("Warning: llm_model provided but AutoTool not found. Mock LLM calls will be used.")
        self.memory = TemporaryMemory()
        self.goal = None
        self.init_obs = None
        self.use_parser = use_parser
        self.debug = debug
        self.task_type = task_type
        # --- State Tracking ---
        self.steps = 0
        self.think_count = 0
        self.max_think_iters = max_think_iters
        self.max_steps = max_steps # Note: Enforcement is typically in the outer loop
        self.round_llm_duration = 0.0
        self.round_cost = 0.0 # Placeholder for round cost (if needed)
        self.token_counts = {"input_tokens": 0, "output_tokens": 0} # Placeholder for token counts
        self.record_mod = record_mod
        # --- Logging & History ---
        self.tool_sequence = [] # Stores parsed tool names if parsing succeeds
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"{self.task_type}_react_baseline_log_{time.strftime('%Y%m%d_%H%M%S')}.json")
        self.action_log = { # Overall log structure for the JSON file
            "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_sequences": 0,
            "sequences": [] # List to hold individual trajectory logs
        }
        self.current_sequence: Optional[Dict[str, Any]] = None # Holds data for the current trajectory

        # --- Statistics Tracking Variables ---
        self.cumulative_time = 0.0

        # --- Prompting Components ---
        self.check_actions_cmd = check_actions
        self.check_inventory_cmd = check_inventory
        self.examples = examples
        self.system_base = system_message
        print(f"2004Examples: {self.examples}")
        self.instruction = instruction
        print(f'2004Instruction: {self.instruction}')
        # --- Initialize toolkit and tools ---
        # self.toolkit = toolkit or Toolkit({})  # Initialize empty toolkit
        # self.toolkit.add_tool(json.loads('/data/agentboard/prompts/ReactBaselineAgent/tool_description.json'))
        # Load examples/system message from file if provided
        print(f"Initializing ReactBaselineAgent with task type: {self.task_type}")
        print(f"init_prompt_path: {init_prompt_path}")
        if init_prompt_path is not None:
            print(f"Loading initialization prompt from file: {init_prompt_path}")
            try:
                with open(init_prompt_path, 'r', encoding='utf-8') as f:
                    self.init_prompt_dict = json.load(f)
                self.examples = self.init_prompt_dict.get("examples", self.examples)
                print(f"Examples loaded: {self.examples}")
                self.system_base = self.init_prompt_dict.get("system_msg", self.system_base)
                self.instruction = self.init_prompt_dict.get("instruction", "")
                print("Prompt loaded successfully.")
            except Exception as e:
                print(f"Error loading prompt file {init_prompt_path}: {e}.")
        # Format examples for the prompt
        if isinstance(self.examples, list):
            print(f'[INFO] Examples provided as list: {self.examples}')
            self.examples_str = "\n---\n".join(self.examples)
        elif isinstance(self.examples, str):
            print(f'[INFO] Examples provided as string: {self.examples}')
            self.examples_str = self.examples
        else:
            self.examples_str = "[No examples provided]"

    def _make_sys_prompt(self, tool_description=""):
        if self.task_type == 'alfworld':
            system_prompt_content = ALFWORLD_SYS_PROMPT.format(
                system_base=self.system_base, goal=self.goal,
                check_actions_cmd=self.check_actions_cmd, 
                check_inventory_cmd="inventory",
                examples_str=self.examples_str
            )
        elif self.task_type in ['tool_query', 'tool-query', 'tool-operation']:
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

    def _parse_raw_action(self, action: str) -> Optional[Dict[str, Any]]:
        parsed_action_result = None
        if self.task_type == "alfworld":
            print(f"Debug: Parsing action: {action}")
            parsed_action_result = parse_alfworld_action(action)
        elif self.task_type == "tool_query":
            parsed_action_result = parse_tool_query(action)
            print(f"Parsed Action: {parsed_action_result}")
        elif self.task_type == "scienceworld":
            parsed_action_result = parse_scienceworld_action(action)
            print(f"Parsed Action: {parsed_action_result}")
        else: 
            parsed_action_result = parse_tool_query(action)
            print(f"Unknown task type. Using default parser: {parsed_action_result}")
        return parsed_action_result

    def reset(self, goal, init_obs):
        """Reset agent state, memory, and start a new trajectory log."""
        # Reset token counters if used
        # self.cumulative_input_tokens = 0
        # self.cumulative_output_tokens = 0
        # self.cumulative_total_tokens = 0

        # --- Initialize logging for the new trajectory ---
        start_time = datetime.now()
        # If there was a previous sequence, save it before starting a new one
        if self.current_sequence:
            self._finalize_and_log_trajectory()
        
        # --- clear attributes for new trajectory ---
        self.memory.clear_memory()
        self.goal = goal
        self.init_obs = init_obs # Store initial observation
        self.steps = 0
        self.think_count = 0
        self.tool_sequence = [] # Reset tool sequence for the new trajectory
        self.cumulative_time = 0.0 # Reset task timer
        # Start the new sequence log
        self.current_sequence = {
            "trajectory_id": len(self.action_log['sequences']), # Simple incremental ID
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
            "steps": [ # Store steps as structured dicts
                {"type": "initial",
                 "content": init_obs,
                 "timestamp": start_time.strftime("%Y%m%d_%H%M%S.%f")}
            ],
            "rewards": [], # Optional: For environment rewards
            # "run_stats": [], # To store stats from each agent.run() call
            "summary": None # Final summary (e.g., success/failure reason)
        }

        # --- Set up initial system prompt ---
        system_prompt_content = ""
        # 使用toolkit的get_tool_desc获取工具描述
        if hasattr(self, 'toolkit') and self.toolkit:
            tool_description = get_tool_desc(self.toolkit)
        else:
            # 如果没有toolkit，使用原来的instruction
            tool_description = self.instruction
            # print(f"Warning: No toolkit found. Using instruction as fallback: {self.instruction}")
            # print(f'check examples: {self.examples}')
            
            
        system_prompt_content = self._make_sys_prompt(tool_description)
        self.memory.add_memory(Message(Role.SYSTEM, system_prompt_content))
        # Add the initial observation to memory as the first user message
        self.memory.add_memory(Message(Role.USER, f"<Observation>{init_obs}</Observation>\n\n"))

        print(f"Agent Reset. Goal: {self.goal}")
        print(f"Initial Observation added to memory.")

    def update(self, action: Optional[str] = None, state: Optional[str] = None):
        """
        Updates the agent's state by recording the last action-observation pair.
        This should be called by the external loop after executing the action.
        """
        observation = state
        self.steps += 1
        if not isinstance(self.current_sequence, dict):
            print("Error: current_sequence not initialized correctly in update.")
            return
        
        parsed_action_result = None
        print(f'[debug before parse]: action={action}, use_parser={self.use_parser}')
        if action and self.use_parser:
            parsed_action_result = self._parse_raw_action(action)

        if action is not None and observation is not None:
            step_record = {
                "type": "act_ob_pair",
                "step_id": self.steps-1,
                "action": {
                    "raw_content": action,
                    "parsed_content": parsed_action_result, # Store parsed result (can be None or unknown)
                    "llm_time_cost": self.round_llm_duration
                },
                "observation": {
                    "raw_content": observation
                },
                "token_consumption": self.token_counts.copy(), # Placeholder for token counts if implemented
                "round_time_cost": self.round_cost,
            }
            print(f"Step {self.steps:02} - llm_time_cost: {self.round_llm_duration:.2f}s, round_cost: {self.round_cost:.2f}s")
            self.current_sequence.setdefault("steps", []).append(step_record)

            # Update tool sequence log (using parsed name if available)
            if parsed_action_result and parsed_action_result["tool_name"] != "unknown":
                self.tool_sequence.append(parsed_action_result["tool_name"])
            elif action: # Fallback to raw action if parsing fails or is off
                self.tool_sequence.append(action) # Or use a special marker like f"RAW:{action}"

            # Add the latest observation to memory for the next LLM call
            # Important: Format it as expected by the prompt
            user_prompt_content = f"<Observation>{observation}</Observation>\n\n"
            self.memory.add_memory(Message(Role.USER, user_prompt_content))

        elif observation is not None:
            # This case might happen if only an initial observation is updated
            # Or if the environment provides an observation without a preceding agent action
            print(f"Debug: update called with observation only: {observation[:100]}...")
            user_prompt_content = f"<Observation>{observation}</Observation>\n\n"
            self.memory.add_memory(Message(Role.USER, user_prompt_content))

        else:
            print("Warning: update called without action or observation.")


    def _increment_llm_calls(self):
        """Increment LLM call count for the current trajectory."""
        if isinstance(self.current_sequence, dict) and "metadata" in self.current_sequence:
            self.current_sequence["metadata"]["total_llm_calls"] += 1


    def parse_response_str(self, response_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
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

    def run(self, init_prompt_dict=None) -> Tuple[bool, Optional[str]]:
        """
        Run the agent for one step to decide the next action.
        Returns:
            - bool: Whether the LLM call and parsing were successful in producing an action.
            - Optional[str]: The decided action string, or None if failed.
        """
        # if not self.current_sequence:
        #      print("Error: Agent not reset. Call reset() before run().")
        #      return False, None

        print('=' * 50); print(f"GOAL: {self.goal}"); 
        print(f"Current Step: {self.steps}"); print('=' * 50)

        run_start_time = time.time()
        llm_call_start_time = 0.0
        thought_count_this_step = 0
        final_action: Optional[str] = None
        llm_call_success = False
        parse_success = False
        response_str = "" # Store the raw response for logging

        # --- ReAct Thinking Loop ---
        while thought_count_this_step < self.max_think_iters:
            thought_count_this_step += 1
            # print(f"--- Thinking Iteration {thought_count_this_step} ---")

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
#                 # print(f"[DEBUG INFO]: LLM input is {llm_input_messages}")
                response_str, self.token_counts["input_tokens"], self.token_counts["output_tokens"] = call_model(llm_input_messages)
                self.current_sequence["metadata"]["total_input_token"] += self.token_counts["input_tokens"]
                self.current_sequence["metadata"]["total_output_token"] += self.token_counts["output_tokens"]
                self.round_llm_duration = time.time() - llm_call_start_time
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
                print(f"Step {self.steps:02} (Parse OK) - Think: {thought}")
                print('=' * 15, "  Action  ", '=' * 15); print(f"Action: {action}")
                final_action = action
                self.think_count = 0 # Reset consecutive think counter
                parse_success = True
                break # Exit loop, action decided

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
                    # Continue the loop

        # --- End of Thinking Loop ---
        run_end_time = time.time()
        duration = run_end_time - run_start_time
        self.round_cost = duration # Store round cost for this run
        # --- Record Statistics for this Run Call ---
        # run_stat = {
        #     "step": self.steps,
        #     "iterations_in_step": thought_count_this_step,
        #     "duration_sec": round(duration, 3),
        #     "llm_call_success": llm_call_success,
        #     "parse_success": parse_success,
        #     "raw_llm_response": response_str, # Log the final raw response
        #     "final_action_generated": final_action,
        #     # Add token counts here if available
        # }
        # if isinstance(self.current_sequence, dict) and self.record_mod is True:
        #     self.current_sequence.setdefault("run_stats", []).append(run_stat)
        # else: print("Warning: Could not log run_stat, current_sequence invalid.")

        # Update cumulative stats for the entire trajectory
        self.cumulative_time += duration
        # Update cumulative token counts if implemented

        print('=' * 17, '  Run End  ', '=' * 17)
        print(f"Run Time: {duration:.2f}s")
        if final_action: print(f"Action Decided: {final_action}")
        else: print(f"No action decided this step. LLM Success: {llm_call_success}, Parse Success: {parse_success}")
        print('=' * 50)

        # Return success status (did we get an action?) and the action itself
        return (final_action is not None), final_action

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
        self.current_sequence['cumulative_llm_time'] = round(self.cumulative_time, 3)
        # Add final token counts if tracked

        self.current_sequence["summary"] = summary

        # Append the completed sequence to the main log list
        self.action_log.setdefault("sequences", []).append(self.current_sequence)
        self.action_log["total_sequences"] = len(self.action_log["sequences"])
        self.action_log["processed_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save the entire log file
        self._save_action_log_file()

        # Clear current_sequence for the next run
        self.current_sequence = None

    def _save_action_log_file(self):
        """Saves the entire action log dictionary to the JSON file."""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.action_log, f, indent=2, ensure_ascii=False)
            # print(f"Action log saved to {self.log_file}")
        except Exception as e:
            print(f"Error saving log file {self.log_file}: {e}")

    # --- Helper Methods ---
    def get_action_sequence(self) -> List[str]:
        """Returns the sequence of tool names executed in the current trajectory."""
        return self.tool_sequence

    def get_current_trajectory_log(self) -> Optional[Dict[str, Any]]:
         """Returns the log data for the currently active trajectory."""
         return self.current_sequence

    def finalize_and_save(self, status="Manually Finalized", summary="Logging explicitly finalized."):
         """Manually finalize the current trajectory and save the log file."""
         print("Finalizing and saving logs...")
         self._finalize_and_log_trajectory(status=status, summary=summary)
         print(f"Log saved to {self.log_file}")

    @classmethod
    def from_config(cls, llm_model, config):
        """Create agent from configuration."""
        tool_env = "tool_query"
        # For tool tasks, attempt to use environment-specific prompt
        default_path = f"/data/agentboard/prompts/ReactBaselineAgent/{tool_env}_sys_prompt.py"
        # init_prompt_path = 'agentboard/prompts/ReactBaselineAgent/tool_query_sys_prompt.py'
        init_prompt_path = config.get("init_prompt_path", default_path)
        print(f"Using prompt file: {init_prompt_path}")
        
        return cls(
            llm_model=llm_model, # LLM model is usually passed separately
            task_type=config.get("task_type", "scienceworld"), # For future use, maybe for different environments
            memory_size=config.get("memory_size", 100),
            examples=config.get("examples", []),
            instruction=config.get("instruction", ""), # May not be used directly by this prompt structure
            init_prompt_path=config.get("init_prompt_path", '/data/agentboard/prompts/ReactBaselineAgent/academic_prompt.json'),
            system_message=config.get("system_message", "You are a helpful assistant."),
            need_goal=config.get("need_goal", True), # Should likely remain True
            check_actions=config.get("check_actions", "check valid actions"),
            check_inventory=config.get("check_inventory", "inventory"),
            use_parser=config.get("use_parser", True),
            max_think_iters=config.get("max_think_iters", 3),
            max_steps=config.get("max_steps", 50),
            debug=config.get("debug", False),
            record_mod=config.get("record_mod", True),
            # toolkit=config.get("toolkit", None) # Add toolkit parameter
        )
    
