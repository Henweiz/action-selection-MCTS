from networks.cnn import CNNPolicyNetwork, CNNValueNetwork
from agents.agent import Agent
import jax
import jax.numpy as jnp
import optax
from networks.network import PolicyNetwork, ValueNetwork
from flax.training import train_state

class AgentGrid(Agent): 
    def __init__(self, params, env):
        params["policy_network"] = CNNPolicyNetwork
        params["value_network"] = CNNValueNetwork
        # TODO: Think those are not really needed. Check if we can remove it. 

        self.key = jax.random.PRNGKey(params['seed'])
        self.env = env        
        _, self.timestep = jax.jit(env.reset)(self.key)
        self._observation_spec = env.observation_spec
        self._action_spec = env.action_spec

        self.policy_network = params.get("policy_network")(num_actions=self._action_spec.num_values, num_channels=32)
        self.value_network = params.get("value_network")(num_channels=32)
        self.policy_optimizer = optax.adam(params['lr'])
        self.value_optimizer = optax.adam(params['lr'])

        input_shape = self.input_shape(self._observation_spec)
        #print(input_shape)

        key1, key2 = jax.random.split(self.key)
        
        self.policy_train_state = train_state.TrainState.create(
            apply_fn=self.policy_network.apply,
            params=self.policy_network.init(key1, jnp.ones((1, *input_shape))),
            tx=self.policy_optimizer
        )

        self.value_train_state = train_state.TrainState.create(
            apply_fn=self.value_network.apply,
            params=self.value_network.init(key2, jnp.ones((1, *input_shape))),
            tx=self.value_optimizer
        )

        self.policy_apply_fn = jax.jit(self.policy_train_state.apply_fn)
        self.value_apply_fn = jax.jit(self.value_train_state.apply_fn)
        self.policy_grad_fn = jax.value_and_grad(self.compute_policy_loss)
        self.value_grad_fn = jax.value_and_grad(self.compute_value_loss)


    def input_shape(self, observation_spec):
        return observation_spec.grid.shape

    def get_state_from_observation(self, observation, batched=True):
        state = observation.grid
        if batched and len(state.shape) == 3:
            state = state[None, ...]
        return state