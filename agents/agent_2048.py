from networks.network import PolicyNetwork, ValueNetwork
from agents.agent import Agent

class Agent2048(Agent): 
    def __init__(self, params, env):
        params["policy_network"] = PolicyNetwork
        params["value_network"] = ValueNetwork
        super().__init__(params, env)

    def input_shape(self, observation_spec):
        return observation_spec.board.shape

    def get_state_from_observation(self, observation, batched=True):
        state = observation.board
        if batched and len(state.shape) == 2:
            state = state[None, ...]
        return state