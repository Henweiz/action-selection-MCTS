import jax
import jax.numpy as jnp
import optax
from networks.network import PolicyNetwork, ValueNetwork
from flax.training import train_state

class Agent: 
    def __init__(self, params, env):
        self.key = jax.random.PRNGKey(params['seed'])
        self.env = env

        # TODO: Think those are not really needed. Check if we can remove it. 
        _, self.timestep = jax.jit(env.reset)(self.key)
        self._observation_spec = env.observation_spec
        self._action_spec = env.action_spec

        self.policy_network = params.get("policy_network", PolicyNetwork)(num_actions=self._action_spec.num_values)
        self.value_network = params.get("value_network", ValueNetwork)()
        self.policy_optimizer = optax.adam(params['lr'])
        self.value_optimizer = optax.adam(params['lr'])

        self.input_shape = self.input_shape_fn(self._observation_spec)

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
        raise NotImplementedError()

    def get_state_from_observation(self, observation, batched):
        raise NotImplementedError()

    # KL Loss between the mcts target & the policy network.
    def compute_policy_loss(self, params, states, actions):
        # Get the probabilities from the policy network
        probs = self.policy_apply_fn(params, states)
        
        # Add epsilon to avoid log(0)
        epsilon = 1e-9
        
        # Compute the KL divergence
        kl_divergence = jnp.sum(actions * jnp.log((actions + epsilon) / (probs + epsilon)), axis=-1)
        mean_loss = jnp.mean(kl_divergence)
        
        return mean_loss

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

    def get_actions(self, state):
        mask = state.action_mask

        # the state has to be gotten depending on the environment
        state = self.get_state_from_observation(state, True)

        # forward pass of the policy network
        actions = self.policy_apply_fn(self.policy_train_state.params, state)
        actions = jnp.ravel(actions)

        # mask and renormalize the actions
        masked_actions = self.mask_actions(actions, mask)
        renormalized_actions = masked_actions / jnp.sum(masked_actions)

        return renormalized_actions
        
    def get_value(self, state):
        # TODO do transform with the value here?

        # the state has to be gotten depending on the environment
        state = self.get_state_from_observation(state, True)

        value = self.value_apply_fn(self.value_train_state.params, state)
        value = jnp.ravel(value)[0]

        return value
    
    def mask_actions(self, actions, mask):
        return jnp.where(mask, actions, 0)

