import jax
import jax.numpy as jnp
import flashbax as fbx
from flax.training import train_state
from flax import linen as nn
import optax
from agent import AlphaZero
import jumanji
from jumanji.wrappers import AutoResetWrapper

params = {
    'env': 'Game2048-v1',
    'seed': 42,
    'lr': 0.001,
    'num_epochs': 10,
    'num_steps': 5,
    'num_actions': 2,
    'num_outputs': 1,
    'buffer_max_length': 10000,  # Set a large buffer size
    'buffer_min_length': 1000,  # Set minimum transitions before sampling
    'sample_batch_size': 32  # Batch size for sampling from the buffer
}

def ce_loss(logits, actions, advantages):
    """ Cross-entropy loss function for the policy network. """
    return -jnp.mean(advantages * jax.nn.log_softmax(logits)[jnp.arange(actions.shape[0]), actions])

def mse_loss(predicted_values, true_values):
    """ Mean squared error loss function for the value network. """
    return jnp.mean((predicted_values - true_values) ** 2)

'''
class PolicyNetwork(nn.Module):
    """A simple policy network that outputs a probability distribution over actions."""
    num_actions: int = 4  # Number of possible actions
    input_shape: int = 4  # Number of inputs from the environment

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.input_shape)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return nn.softmax(x)


class ValueNetwork(nn.Module):
    """A simple value network."""
    num_outputs: int = 1
    input_shape: int = 4 

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(8)(x)
        # x = nn.relu(x)
        # x = nn.Dense(8)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_outputs)(x)
        return x

class TrainState(train_state.TrainState):
    pass


class AlphaZero:
    def __init__(self):
        self.policy_network = PolicyNetwork(params['num_actions'])
        self.value_network = ValueNetwork()
        self.policy_optimizer = optax.adam(params['lr'])
        self.value_optimizer = optax.adam(params['lr'])

        self.policy_train_state = TrainState.create(
            apply_fn=self.policy_network.apply,
            params=self.policy_network.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 10)))['params'],
            tx=self.policy_optimizer,
        )

        self.value_train_state = TrainState.create(
            apply_fn=self.value_network.apply,
            params=self.value_network.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 10)))['params'],
            tx=self.value_optimizer,
        )

        # Set up the Flashbax buffer
        self.buffer = fbx.make_flat_buffer(
            max_length=params['buffer_max_length'],
            min_length=params['buffer_min_length'],
            sample_batch_size=params['sample_batch_size']
        )
        # Initialize the buffer's state
        fake_initial_data = {"obs": jnp.zeros((10, 10)), "reward": jnp.array(0.0)}
        self.buffer_state = self.buffer.init(fake_initial_data)

    def train_step(self, rng_key):
        if self.buffer.can_sample(self.buffer_state):
            batch = self.buffer.sample(self.buffer_state, rng_key)
            states = batch.experience.first['obs']
            next_states = batch.experience.second['obs']
            rewards = batch.experience.second['reward']

            def policy_loss_fn(params):
                logits = self.policy_network.apply({'params': params}, states)
                # Your logic to compute advantages will go here
                advantages = jnp.zeros(logits.shape[0])  # Placeholder
                return ce_loss(logits, batch['actions'], advantages)

            def value_loss_fn(params):
                values = self.value_network.apply({'params': params}, states)
                return mse_loss(values, rewards)

            policy_grads = jax.grad(policy_loss_fn)(self.policy_train_state.params)
            value_grads = jax.grad(value_loss_fn)(self.value_train_state.params)

            self.policy_train_state = self.policy_train_state.apply_gradients(grads=policy_grads)
            self.value_train_state = self.value_train_state.apply_gradients(grads=value_grads)

    def run_episode(self, rng_key):
        # You need to replace this with actual game logic that records observations, rewards, etc.
        pass

    def train(self):
        rng_key = jax.random.PRNGKey(0)
        for epoch in range(params['num_epochs']):
            rng_key, subkey = jax.random.split(rng_key)
            self.run_episode(subkey)
            self.train_step(rng_key)

'''

def main(unused_arg):
    env = jumanji.make(params['env'])
    env = AutoResetWrapper(env)
    params['num_actions'] = env.action_spec.num_values
    agent = AlphaZero(params, env)


