import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import grad, jit
from flax.training import train_state
import optax


class RandomNetwork_2048(nn.Module):
    num_actions: int
    num_channels: int

    @nn.compact
    def __call__(self, x):
        # Conv Layers + MLP Layer.

        # randomize
        actions = jnp.ones((x.shape[0], self.num_actions))

        value = jnp.ones((x.shape[0], 1))

        return actions, value