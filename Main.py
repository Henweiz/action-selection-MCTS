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
    'num_actions': 4,
    'buffer_max_length': 10000,  # Set a large buffer size
    'buffer_min_length': 1000,  # Set minimum transitions before sampling
    'sample_batch_size': 32  # Batch size for sampling from the buffer
}

def main(unused_arg):
    env = jumanji.make(params['env'])
    env = AutoResetWrapper(env)
    params['num_actions'] = env.action_spec.num_values
    agent = AlphaZero(params, env)


