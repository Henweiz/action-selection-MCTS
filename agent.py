import collections
import random
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import flashbax as fbx
from network import PolicyNetwork, ValueNetwork
from flax.training import train_state

class AlphaZero: 
    def __init__(self, params, env):
        self.key = jax.random.PRNGKey(params['seed'])
        self.env = env

        self.state, self.timestep = jax.hit(env.reset)(self.key)
        self._observation_spec = env.observation_spec
        self._action_spec = env.action_spec
        self._target_period = env.target_period
        # Neural net and optimiser.
        self.policy_network = PolicyNetwork(num_actions=self._action_spec.num_values)
        self.value_network = ValueNetwork()
        self.policy_optimizer = optax.adam(params['lr'])
        self.value_optimizer = optax.adam(params['lr'])

        input_shape = self._observation_spec.shape

        key1, key2 = jax.random.split(self.key)
        
        self.policy_train_state = train_state.TrainState.create(
            apply_fn=self.policy_network.apply,
            params=self.policy_network.init(key1, jnp.ones((1, *input_shape)))['params'],
            tx=self.policy_optimizer
        )

        self.value_train_state = train_state.TrainState.create(
            apply_fn=self.value_network.apply,
            params=self.value_network.init(key2, jnp.ones((1, *input_shape)))['params'],
            tx=self.value_optimizer
        )

        # Set up the Flashbax buffer
        self.buffer = fbx.make_flat_buffer(
            max_length=params['buffer_max_length'],
            min_length=params['buffer_min_length'],
            sample_batch_size=params['sample_batch_size']
        )
        # Initialize the buffer's state
        fake_initial_data = {"obs": jnp.zeros(input_shape), "reward": jnp.array(0.0)}
        self.buffer_state = self.buffer.init(fake_initial_data)

        self.policy_apply_fn = jax.jit(self.policy_train_state.apply_fn)
        self.value_apply_fn = jax.jit(self.value_train_state.apply_fn)
        self.policy_grad_fn = jax.value_and_grad(self.compute_policy_loss)
        self.value_grad_fn = jax.value_and_grad(self.compute_value_loss)

    def compute_policy_loss(self, params, observations, actions, advantages):
        logits = self.policy_apply_fn(params, observations)
        action_probs = jnp.take_along_axis(logits, actions[:, None], axis=-1).squeeze()
        loss = -jnp.mean(jnp.log(action_probs) * advantages) #
        return loss

    def compute_value_loss(self, params, observations, returns):
        values = self.value_apply_fn(params, observations)
        loss = jnp.mean((returns - values.squeeze()) ** 2) #MSE Loss
        return loss

    def update_policy(self, observations, actions, advantages):
        loss, grads = self.policy_grad_fn(self.policy_train_state.params, observations, actions, advantages)
        self.policy_train_state = self.policy_train_state.apply_gradients(grads=grads)
        return loss

    def update_value(self, observations, returns):
        loss, grads = self.value_grad_fn(self.value_train_state.params, observations, returns)
        self.value_train_state = self.value_train_state.apply_gradients(grads=grads)
        return loss

    def train(self, timestep):
        self.buffer = self.buffer.add(self.state, timestep)
        if self.buffer.can_sample(self.buffer_state):
            batch = self.buffer.sample(self.buffer_state, self.key)
            states = batch.experience.first['obs']
            actions = batch.experience.second['action']
            rewards = batch.experience.second['reward']
            next_states = batch.experience.second['obs'] # Not used atm
            
            # Calculate returns (May need to change)
            returns = rewards 
            
            value_loss = self.update_value(states, returns)

            # Compute advantages (Not sure if we need to compute advantages, TODO: Look into Alphazero loss functions)
            advantages = returns - value_loss

            policy_loss = self.update_policy(states, actions, advantages)
            
            return policy_loss, value_loss

    def select_action(self, observation):
        logits = self.policy_apply_fn(self.policy_train_state.params, observation[None])
        action_probs = jnp.squeeze(logits)
        action = jax.random.choice(jax.random.PRNGKey(np.random.randint(1e6)), a=len(action_probs), p=action_probs)
        return action
    
    def take_action(self, observation):
        action = self.select_action(observation)
        new_state, timestep = self.env.step(self.state, action)

        return new_state, timestep