from common.registry import registry
from .react_agent import ReactBaselineAgent
from .reflection_agent import ReflectionAgent
from .test_agent import ReactInertiaAgent
from .test_agent2 import ReflectionInertialAgent 
from .test_agent3 import ReactNgramAgent
__all__ = ["ReactBaselineAgent", "ReactInertiaAgent", "ReflectionAgent", "ReflectionInertialAgent", "ReactNgramAgent"]


def load_agent(name, config, llm_model):
    agent = registry.get_agent_class(name).from_config(llm_model, config)
    print(f'load agent done, the agent is {agent.__class__.__name__}')
    return agent
