import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import grad, jit
from flax.training import train_state
import optax
from jax import random

# TODO is this still used or can it be removed?
class CNNPolicyNetwork(nn.Module):
    """A simple policy network that outputs a probability distribution over actions."""

    num_actions: int
    num_channels: int

    @nn.compact
    def __call__(self, x):
        #x = jnp.reshape(x, (x.shape[0], -1)) #flatten, do not that we get errors when we do not input batches
        k_size = (3, 3)

        x = nn.Conv(features=self.num_channels, kernel_size=k_size)(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=self.num_channels, kernel_size=k_size)(x)
        x = nn.leaky_relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten
        x = nn.Dense(64)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(64)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(64)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(self.num_actions)(x)
        x = nn.softmax(x)
        return x


class CNNValueNetwork(nn.Module):
    """A simple value network."""
    num_channels: int

    @nn.compact
    def __call__(self, x):
        k_size = (3, 3)

        x = nn.Conv(features=self.num_channels, kernel_size=k_size)(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(features=self.num_channels, kernel_size=k_size)(x)
        x = nn.leaky_relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(1)(x)
        return x
