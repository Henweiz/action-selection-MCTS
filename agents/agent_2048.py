from networks.network_2048 import PolicyValueNetwork_2048
from agents.agent import Agent
import jax.numpy as jnp

class Agent2048(Agent): 
    def __init__(self, params):
        params["network"] = PolicyValueNetwork_2048
        super().__init__(params)


    def normalize_rewards(self, r):
        # Log2 the rewards and then normalize given that 2^16 is the highest realistic reward
        return jnp.log2(r + 1) / 16

    def reverse_normalize_rewards(self, r):
         # do the reverse of the above
        return 2 ** (r * 16) - 1

    def input_shape_fn(self, observation_spec):
        return observation_spec.board.shape

    def get_state_from_observation(self, observation, batched=True):
        state = observation.board
        if batched and len(state.shape) == 2:
            state = state[None, ...]
        return state