from networks.network import PolicyValueNetwork
from agents.agent import Agent

class Agent2048(Agent): 
    def __init__(self, params, env):
        params["network"] = PolicyValueNetwork
        super().__init__(params, env)

    def input_shape(self, observation_spec):
        return observation_spec.board.shape

    def get_state_from_observation(self, observation, batched=True):
        state = observation.board
        if batched and len(state.shape) == 2:
            state = state[None, ...]
        return state