from networks.network_2048 import PolicyValueNetwork_2048
from agents.agent import Agent

class Agent2048(Agent): 
    def __init__(self, params):
        params["network"] = PolicyValueNetwork_2048
        super().__init__(params)


    def normalize_rewards(self, r):
        # Makes the mse loss be in about the same range as the KL loss
        return r / 50

    def reverse_normalize_rewards(self, r):
         # Reverse the above normalization
        return r * 50

    def input_shape_fn(self, observation_spec):
        return observation_spec.board.shape

    def get_state_from_observation(self, observation, batched=True):
        state = observation.board
        if batched and len(state.shape) == 2:
            state = state[None, ...]
        return state