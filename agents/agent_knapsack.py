import jax.numpy as jnp
from agents.agent import Agent
from networks.network_knapsack import forward_fn
import haiku.experimental.flax as hkflax
import haiku as hk
import jax
from flax.training import train_state
import optax

class AgentKnapsack(Agent):
    def __init__(self, params):
        self.network = hkflax.Module(hk.transform(forward_fn))
        self.optimizer = optax.adam(params['lr'])
        self.input_shape = self.input_shape_fn(params["obs_spec"])

        self.key = jax.random.PRNGKey(params['seed'])
        
        self.train_state = train_state.TrainState.create(
            apply_fn=self.network.apply,
            params=self.network.init(self.key, jnp.ones((1, *self.input_shape))),
            tx=self.optimizer
        )

        self.net_apply_fn = jax.jit(self.train_state.apply_fn)
        self.grad_fn = jax.value_and_grad(self.loss_fn)
    
        self.last_mse_losses = []
        self.last_kl_losses = []
    # def mask_actions(self, actions, mask):
    #     return actions

    def input_shape_fn(self, observation_spec):
        return (4, *observation_spec.weights.shape)

    def get_state_from_observation(self, observation, batched=True):
        state = jnp.stack([observation.weights, observation.values, observation.packed_items, observation.action_mask], axis=-2)
        assert state.shape[-2:] == (4, observation.weights.shape[-1])
        if batched and len(state.shape) == 2:
            state = state[None, ...]
        return state