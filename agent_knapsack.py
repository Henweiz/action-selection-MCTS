import jax.numpy as jnp
from knapsack_networks import KnapsackPolicyNetwork, KnapsackValueNetwork
from agent import Agent

class AgentKnapsack(Agent):
    def __init__(self, params, env):
        params["policy_network"] = KnapsackPolicyNetwork
        params["value_network"] = KnapsackValueNetwork
        super().__init__(params, env)

    def input_shape(self, observation_spec):
        return (4, *observation_spec.weights.shape)

    def get_state_from_observation(self, observation, batched=True):
        state = jnp.stack([observation.weights, observation.values, observation.packed_items, observation.action_mask], axis=-2)
        assert state.shape[-2:] == (4, observation.weights.shape[-1])
        if batched and len(state.shape) == 2:
            state = state[None, ...]
        return state