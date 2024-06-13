from networks.cnn import CNNPolicyNetwork, CNNValueNetwork
from agents.agent import Agent
import jax
import jax.numpy as jnp
import optax
from networks.network import PolicyValueNetwork
from flax.training import train_state

class AgentGrid(Agent): 
    def __init__(self, params):
        params["network"] = PolicyValueNetwork
        super().__init__(params)


    def input_shape_fn(self, observation_spec):
        return observation_spec.grid.shape

    def get_state_from_observation(self, observation, batched=True):
        state = observation.grid
        if batched and len(state.shape) == 3:
            state = state[None, ...]
        return state