import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import grad, jit
from flax.training import train_state
import optax


class PolicyValueNetwork_2048(nn.Module):
    num_actions: int
    num_channels: int

    @nn.compact
    def __call__(self, x):
        # Conv Layers + MLP Layer.

        k_size = (2, 2)
        x = nn.Conv(features=self.num_channels, kernel_size=k_size)(x)
        x = nn.leaky_relu(x)
        x = jnp.reshape(x, (x.shape[0], -1))  # Flatten

        # Policy Layers.
        actions = nn.Dense(128)(x) # TODO try value norm
        actions = nn.leaky_relu(actions)
        actions = nn.Dense(128)(actions)
        actions = nn.leaky_relu(actions)
        actions = nn.Dense(self.num_actions)(actions)
        actions = nn.softmax(actions)


        # Value Layers
        value = nn.Dense(256)(x)
        value = nn.leaky_relu(value)
        value = nn.Dense(256)(value)
        value = nn.leaky_relu(value)
        value = nn.Dense(1)(value)

        return actions, value