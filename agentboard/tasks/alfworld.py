import json
from agents import load_agent
from environment import load_environment
from common.registry import registry
import copy
import os
from datetime import datetime

from utils.logging.logger import TaskLogger
from utils.logging.agent_logger import AgentLogger
logger = AgentLogger(__name__)


from .base_task import BaseTask


prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}



@registry.register_task("alfworld")
class Evalalfworld(BaseTask):
    def __init__(self,
                 llm_config=None,
                 agent_name='agent_name',
                 max_num_steps=30,
                 num_exams=134,
                 init_prompt_path='prompts/alfworld_base.json',
                 agent_config=None,
                 env_config=None,
                 llm = None,
                 baseline_dir = None,
                 log_path = None
                 ):
        
        super().__init__()
        
        ####################  initialize agent ##################
        self.agent = load_agent(agent_name, agent_config, llm)
        #################################################################
        
        with open(init_prompt_path, 'r') as f:
            self.prompts = json.load(f)
        self.env_cfg = env_config
        self.max_num_steps = max_num_steps
        self.num_exams = num_exams
        
        self.baseline_dir = baseline_dir
        
        
        self.agentboard = TaskLogger(task_name="alfworld", log_path=log_path, max_num_steps=self.max_num_steps, baseline_dir=self.baseline_dir)
        
        # Append 添加日志相关的初始化
        self.action_sequences = {
            "processed_time": None,
            "total_sequences": 0,
            "total_tool_calls": 0,
            "sequences": []
        }
        self.current_sequence = None

    def parseAction(self, action):
        action = action.strip()
        if "put" in action:
            if " in " in action:
                action = action.replace(" in ", ' in/on ')
            elif " on " in action:
                action = action.replace(" on ", ' in/on ')
        if action.endswith('.'):
            action = action[:-1].strip()
        return action

    def evaluate_env(self,  index, ob='', examples=None):
        # Append 初始化当前序列
        self.current_sequence = {
            "task_id": index,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "task_type": self.env.cur_task_name.split('/')[0],
            "action_sequence": [],
            "observations": [],
            "rewards": []
        }

        init_ob = ob.split('\n')[0]
        goal = ob.split('\n')[1].split("Your task is to:")[1].strip()
        print(f'start reset agent')
        self.agent.reset(goal=goal, init_obs=init_ob)
        print(f'agent reset done')
        logger.goal("Example {} | Goal: {}".format(index, self.agent.goal))
        init_prompt_dict = copy.deepcopy(self.prompts)
        init_prompt_dict['examples'] = examples
        reward = 0.
        last_reward = 0.
        done = False
        grounding_acc_count = 0
        score_change_record = []
        logger.info("Step {:02} - Message: {}".format(0, init_ob))
        
        trajectory = []
        trajectory.append({"Goal":goal, "id":0})
        trajectory.append({"Observation":init_ob, "id":0})   
        
        for i in range(0, self.max_num_steps):
            print(f'start run agent')
            success, action = self.agent.run(init_prompt_dict=init_prompt_dict)
            print(f'success: {success}, action: {action}')
            print(f'run agent done')    
            if not success:
                break
            
            action = self.parseAction(action)
            if action in self.env.get_action_space():
                grounding_acc_count += 1.0
            
            logger.info("Step {:02} - Action: {}".format(i, action))
            
            # Append 记录action
            self.current_sequence["action_sequence"].append(action)
            
            trajectory.append({"Action":action, "id":i})
            
            observation, reward, done, info = self.env.step(action)
            logger.info("Step {:02} - Observation: {}".format(i, observation))

            # 移到这里，env.step之后
            self.current_sequence["observations"].append(observation)
            self.current_sequence["rewards"].append(reward)

            if "Task accomplished!" in observation and reward < 1.0:
                raise Exception("Task accomplished error")
            
            logger.info("Step {:02} - Progress Rate: {}\n".format(i, reward))
            
            trajectory.append({"Observation":observation, "id":i})
            trajectory.append({"Progress Rate":reward, "id":i})
            
            #print(f'Step: {str(i)} Action: {action}\nObservation: {observation}')
            #print(f"reward: {reward}, isdone: {done}")
            
            if reward > last_reward:
                score_change_record.append((i, reward))
            last_reward = reward
            self.agent.update(action=action, state=observation)
            if done:
                
                game_name = self.env.cur_task_name.split('/')[0]
                env_details = {"task_name": game_name, "goal": self.agent.goal, "difficulty": self.env.difficulty}
                self.agentboard.log_example(index, True, reward, grounding_acc_count / (i + 1), score_change_record, env_details, trajectory)
                    
                # Append 序列结束时保存
                self.current_sequence["final_reward"] = reward
                self.current_sequence["success"] = done
                self.current_sequence["steps"] = i + 1
                self.action_sequences["sequences"].append(self.current_sequence)
                
                # --- MODIFICATION: Ensure log is saved on successful completion ---
                if hasattr(self.agent, '_save_action_log') and callable(self.agent._save_action_log):
                    self.agent._save_action_log() # 确保agent是具有这些方法
                else:
                    print("Warning: Agent does not have _save_action_log method, cannot save log.")
                return 1.0, True, grounding_acc_count / (i + 1), score_change_record, i

        
        game_name = self.env.cur_task_name.split('/')[0]
        env_details = {"task_name": game_name, "goal": self.agent.goal, "difficulty": self.env.difficulty}
        
        
        progress_rate = reward 
        
        try: example_prompt = self.agent.get_example_prompt()
        except: example_prompt = None  
        self.agentboard.log_example(index, done, progress_rate, grounding_acc_count / (i + 1), score_change_record, env_details, trajectory, example_prompt)

        # Append 序列结束时保存
        if self.current_sequence:
            self.current_sequence["final_reward"] = reward
            self.current_sequence["success"] = done
            self.current_sequence["steps"] = i + 1
            self.action_sequences["sequences"].append(self.current_sequence)
            # --- MODIFICATION: Ensure log is saved even with max steps exceeded ---
            if hasattr(self.agent, '_save_action_log') and callable(self.agent._save_action_log):
                self.agent._save_action_log()
            else:
                print("Warning: Agent does not have _save_action_log method, cannot save log.")

            self.action_sequences["sequences"].append(self.current_sequence)
        return progress_rate, done, grounding_acc_count / (i + 1), score_change_record, i

    def evaluate(self):
        # Append 初始化日志时间
        self.action_sequences["processed_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.env = load_environment('alfworld', self.env_cfg)
        scores = []
        score_state_records = []
        grounding_accs = []
        srs = []
        difficulties = []

        for id in range(self.num_exams):

            ob, info = self.env.reset()
            ob = '\n'.join(ob[0].split('\n\n')[1:])
            name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
            #sub_goal = selected_obs[name]
            difficulties.append(self.env.difficulty)

            for i, (k, v) in enumerate(prefixes.items()):
                if name.startswith(k):
                    examples = "".join(self.prompts['examples'][v])
                    score, is_done, grounding_acc, score_change_record, steps = self.evaluate_env(ob=ob, examples=examples, index=id)
                    if is_done:
                        srs.append(1.0)
                    else:
                        srs.append(0.0)
                    scores.append(score)
                    grounding_accs.append(grounding_acc)
                    score_state_records.append(score_change_record)
                    #print(f"the {i}th task: reward: {score}")
                    logger.finish("Example {} | Success: {} , Progress Rate: {} , Steps: {}\n".format(id, is_done, score, steps))

        sr = sum(srs) * 1.0 / len(srs)
        pr = sum(scores) * 1.0 / len(scores)
        gr = sum(grounding_accs) * 1.0 / len(grounding_accs)

        hard_sr = [sr for sr, difficulty in zip(srs, difficulties) if difficulty == "hard"]
        hard_sr = sum(hard_sr) / len(hard_sr) if len(hard_sr) > 0 else 0

        hard_pr = [pr for pr, difficulty in zip(scores, difficulties) if difficulty == "hard"]
        hard_pr = sum(hard_pr) / len(hard_pr) if len(hard_pr) > 0 else 0

        easy_sr = [sr for sr, difficulty in zip(srs, difficulties) if difficulty == "easy"]
        easy_sr = sum(easy_sr) / len(easy_sr) if len(easy_sr) > 0 else 0

        easy_pr = [pr for pr, difficulty in zip(scores, difficulties) if difficulty == "easy"]
        easy_pr = sum(easy_pr) / len(easy_pr) if len(easy_pr) > 0 else 0
                    
        
        self.agentboard.log_summary(sr, pr, gr, score_state_records, hard_sr, hard_pr, easy_sr, easy_pr)

        # Append 评估结束后保存完整的日志
        self.action_sequences["total_sequences"] = len(self.action_sequences["sequences"])
        self.action_sequences["total_tool_calls"] = sum(len(seq["action_sequence"]) 
                                                       for seq in self.action_sequences["sequences"])
        
        # Append 保存到文件
        if self.log_path:
            log_dir = os.path.dirname(self.log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            action_log_path = os.path.join(log_dir, f'action_sequences_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(action_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.action_sequences, f, indent=2, ensure_ascii=False)
        
        return  srs, scores, grounding_accs, score_state_records, easy_sr, hard_sr, easy_pr, hard_pr

    def _grounding_fn(self, action):

        if action not in self.env.GetValidActions():
            print(f"The wrong action is: {action}")
            return "check valid actions"
        else:
            return action

    @classmethod
    def from_config(cls,
                    run_config,
                    llm_config,
                    agent_config,
                    env_config,
                    llm = None  
                    ):

        agent_name = agent_config.get("name", "GPTAgent")
        init_prompt_path = agent_config.get("init_prompt_path", 'prompts/alfworld_in_context_learning.json') 
        max_num_steps = run_config.get("max_num_steps", 30)
        baseline_dir = run_config.get("baseline_dir", "data/baseline_results")
        # wandb = run_config.get("wandb", False)
        num_exams = run_config.get("num_exam", 134)
        log_path = run_config.get("log_path", None)
        return cls(
                   llm_config=llm_config,
                   agent_name=agent_name,
                   max_num_steps=max_num_steps,
                   num_exams=num_exams,
                   init_prompt_path=init_prompt_path,
                   agent_config=agent_config,
                   env_config=env_config,
                   llm = llm,
                   baseline_dir = baseline_dir,
                   log_path = log_path
                   )