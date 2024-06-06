import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import grad, jit
from flax.training import train_state
import optax


class CNNPolicyNetwork(nn.Module):
    """A simple policy network that outputs a probability distribution over actions."""

    num_actions: int # Number of possible actions
    num_channels: int

    @nn.compact
    def __call__(self, x):
        #x = jnp.reshape(x, (x.shape[0], -1)) #flatten, do not that we get errors when we do not input batches
        x = nn.Conv(features=self.num_channels, kernel_size=(2, 2), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_channels, kernel_size=(2, 2), strides=(1, 1))(x)
        x = nn.relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x


# Define the Value Network
class CNNValueNetwork(nn.Module):
    """A simple value network."""
    #num_outputs: int = 1
    num_channels: int

    @nn.compact
    def __call__(self, x):
        #x = jnp.reshape(x, (x.shape[0], -1)) # flatten
        x = nn.Conv(features=self.num_channels, kernel_size=(2, 2), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.num_channels, kernel_size=(2, 2), strides=(1, 1))(x)
        x = nn.relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
