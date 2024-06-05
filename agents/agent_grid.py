from networks.network import PolicyNetwork, ValueNetwork
from agents.agent import Agent

class AgentGrid(Agent): 
    def __init__(self, params, env):
        params["policy_network"] = PolicyNetwork
        params["value_network"] = ValueNetwork
        super().__init__(params, env)

    def input_shape(self, observation_spec):
        return observation_spec.grid.shape

    def get_state_from_observation(self, observation, batched=True):
        state = observation.grid
        if batched and len(state.shape) == 3:
            state = state[None, ...]
        return state