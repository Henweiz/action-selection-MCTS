import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import grad, jit
from flax.training import train_state
import optax

class PolicyNetwork(nn.Module):
    """A simple policy network that outputs a probability distribution over actions."""

    num_actions: int # Number of possible actions

    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, (x.shape[0], -1)) #flatten, do not that we get errors when we do not input batches
        x = nn.Dense(64)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(64)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.num_actions)(x)
        x = nn.softmax(x)
        return x

# Define the Value Network
class ValueNetwork(nn.Module):
    """A simple value network."""
    #num_outputs: int = 1

    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, (x.shape[0], -1)) # flatten
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(1)(x)
        return x
