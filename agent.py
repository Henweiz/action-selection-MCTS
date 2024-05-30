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

        # TODO: Think those are not really needed. Check if we can remove it. 
        _, self.timestep = jax.jit(env.reset)(self.key)
        self._observation_spec = env.observation_spec
        self._action_spec = env.action_spec
        #print(self._action_spec.num_values)
        # Neural net and optimiser.
        self.policy_network = PolicyNetwork(num_actions=self._action_spec.num_values)
        self.value_network = ValueNetwork()
        self.policy_optimizer = optax.adam(params['lr'])
        self.value_optimizer = optax.adam(params['lr'])

        input_shape = self._observation_spec.board.shape
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

        # Set up the Flashbax buffer
        self.buffer = fbx.make_flat_buffer(
            max_length=params['buffer_max_length'],
            min_length=params['buffer_min_length'],
            sample_batch_size=params['sample_batch_size']
        )
        # Initialize the buffer's state
        self.buffer_state = self.buffer.init(self.timestep)

        self.policy_apply_fn = jax.jit(self.policy_train_state.apply_fn)
        self.value_apply_fn = jax.jit(self.value_train_state.apply_fn)
        self.policy_grad_fn = jax.value_and_grad(self.compute_policy_loss)
        self.value_grad_fn = jax.value_and_grad(self.compute_value_loss)


    # KL Loss between the mcts target & the policy network.
    def compute_policy_loss(self, params, states, actions):
        logits = self.policy_apply_fn(params, states)
        epsilon = 1e-9
        target = jnp.where(actions == 0, epsilon, actions)
        logits = jnp.log(logits)

        kl_loss = jnp.sum(target * (jnp.log(target) - logits))
        kl_loss = jnp.mean(kl_loss)

        return kl_loss

    # MSE Loss between the value network and the returns
    def compute_value_loss(self, params, states, returns):
        values = self.value_apply_fn(params, states)
        loss = jnp.mean((returns - values) ** 2) 
        return loss


    def update_policy(self, states, actions):
        loss, grads = self.policy_grad_fn(self.policy_train_state.params, states, actions)
        self.policy_train_state = self.policy_train_state.apply_gradients(grads=grads)
        return loss

    def update_value(self, states, returns):
        loss, grads = self.value_grad_fn(self.value_train_state.params, states, returns)
        self.value_train_state = self.value_train_state.apply_gradients(grads=grads)
        return loss

    def update(self, timestep, actions):
        self.buffer_state = self.buffer.add(self.buffer_state, timestep)

        if self.buffer.can_sample(self.buffer_state):
            batch = self.buffer.sample(self.buffer_state, self.key).experience
            #print(batch)
            states = batch.first.observation.board
            rewards = batch.second.reward

            # Calculate returns (May need to change)
            returns = rewards
            #values = self.value_apply_fn(self.value_train_state.params, states)
            #print(f"Board states: {states.shape}")
            value_loss = self.update_value(states, returns)

            policy_loss = self.update_policy(states, actions)

            return policy_loss, value_loss

        # Not sure if we need to return 0, 0. Guess that it does not matter...
        return 0, 0





        
    def get_actions(self, state):
        #print(state)
        #todo: there's an issue with batches, first if fixes the input when we get a batchless state
        if len(state.board.shape) == 2:
            state = state.board[None, ...]
        else:
            state = state.board
        actions = self.policy_apply_fn(self.policy_train_state.params, state)
        actions = jnp.ravel(actions)
        # print(f"Actions shape: {actions.shape}")
        return actions
        
    def get_value(self, state):
        #print(state)
        #todo: there's an issue with batches, first if fixes the input when we get a batchless state
        if len(state.board.shape) == 2:
            state = state.board[None, ...]
        else:
            state = state.board
        value = self.value_apply_fn(self.value_train_state.params, state)
        value = jnp.ravel(value)[0]
        #print(f"Values shape: {value.shape}")
        return value



    # def select_action(self, state):
    #
    #     # TODO how to select
    #     return jnp.argmax(self.get_actions(state))

    def update_new(self, states, actions, rewards, next_states, dones):
        # Similar to existing update but takes full batches and processes returns, etc.
        returns = self.compute_returns(rewards, next_states, dones)  # Implement this
        policy_loss = self.update_policy(states, actions)
        value_loss = self.update_value(states, returns)
        return policy_loss, value_loss
