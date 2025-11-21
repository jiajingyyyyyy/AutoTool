from environment.academia_env import AcademiaEnv
from environment.scienceworld_env import Scienceworld
from environment.alfworld.alfworld_env import AlfWorld
print("Loading environment module...")
from agentboard.common.registry import registry
print("Registry loaded.")
import json
import os

# __all__ = [
#     "BabyAI",
#     "AlfWorld",
#     "Scienceworld",
    
#     "PDDL",
#     "Jericho",
    
#     "AcademiaEnv",
#     "MovieEnv",
#     "TodoEnv",
#     "SheetEnv",
#     "WeatherEnv",
    
#     "Webshop",
#     "BrowserEnv",
# ]


def load_environment(name, config):
    
    if name not in registry.list_environments():
        if name == 'babyai': from agentboard.environment.babyai_env import BabyAI
        if name == "academia": from agentboard.environment.academia_env import AcademiaEnv
        if name == "todo": from agentboard.environment.todo_env import TodoEnv
        if name == "jericho": from agentboard.environment.jericho_env import Jericho
        if name == "webshop": from agentboard.environment.webshop_env import Webshop
        if name == "alfworld": from agentboard.environment.alfworld.alfworld_env import AlfWorld
        if name == "scienceworld": from agentboard.environment.scienceworld_env import Scienceworld
    # print(f"Environment {name} loaded with config: {config}")
    print("Available environments:", registry.list_environments())
    print("Requested environment:", name)  # name 应该是 'tool-query' 或 'movie'
    env = registry.get_environment_class(name).from_config(config)
    
    return env

