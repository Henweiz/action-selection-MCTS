from networks.cnn import CNNPolicyNetwork, CNNValueNetwork
from agents.agent import Agent
import jax
import jax.numpy as jnp
import optax
from networks.network import PolicyNetwork, ValueNetwork
from flax.training import train_state

class AgentMaze(Agent): 
    def __init__(self, params, env):
        params["policy_network"] = CNNPolicyNetwork
        params["value_network"] = CNNValueNetwork
        # TODO: Think those are not really needed. Check if we can remove it. 

        self.key = jax.random.PRNGKey(params['seed'])
        self.env = env        
        state, self.timestep = jax.jit(env.reset)(self.key)
        #print(self.timestep.discount)
        self._observation_spec = state
        self._action_spec = env.action_spec

        self.policy_network = params.get("policy_network")(num_actions=self._action_spec.num_values, num_channels=params['num_channels'])
        self.value_network = params.get("value_network")(num_channels=params['num_channels'])
        self.policy_optimizer = optax.adam(params['lr'])
        self.value_optimizer = optax.adam(params['lr'])

        self.input_shape = self.input_shape_fn(self._observation_spec).shape
        print(self.input_shape)

        key1, key2 = jax.random.split(self.key)
        
        self.policy_train_state = train_state.TrainState.create(
            apply_fn=self.policy_network.apply,
            params=self.policy_network.init(key1, jnp.ones((1, *self.input_shape))),
            tx=self.policy_optimizer
        )

        self.value_train_state = train_state.TrainState.create(
            apply_fn=self.value_network.apply,
            params=self.value_network.init(key2, jnp.ones((1, *self.input_shape))),
            tx=self.value_optimizer
        )

        self.policy_apply_fn = jax.jit(self.policy_train_state.apply_fn)
        self.value_apply_fn = jax.jit(self.value_train_state.apply_fn)

        self.policy_grad_fn = jax.value_and_grad(self.compute_policy_loss)
        self.value_grad_fn = jax.value_and_grad(self.compute_value_loss)


    def input_shape_fn(self, observation_spec):
        return self.process_observation(observation_spec)

    def get_state_from_observation(self, observation, batched=True):
        state = self.process_observation(observation)
        if batched and len(state.shape) == 3:
            state = state[None, ...]
        return state
    
    def process_observation(self, observation):
        """Add the agent and the target to the walls array."""
        agent = 2
        target = 3
        obs = observation.walls.astype(float)
        obs = obs.at[tuple(observation.agent_position)].set(agent)
        obs = obs.at[tuple(observation.target_position)].set(target)
        return jnp.expand_dims(obs, axis=-1)  # Adding a channels axis.