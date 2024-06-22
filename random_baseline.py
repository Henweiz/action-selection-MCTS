from Main import get_rewards, params

import jax
import jax.numpy as jnp

import wandb_logging
import jumanji

from jumanji.wrappers import AutoResetWrapper


@jax.jit
def env_step(state, action):
    """A single step in the environment."""
    next_state, next_timestep = env.step(state, action)
    return next_state, next_timestep

def run_n_steps(state, timestep, subkey, n):
    random_keys = jax.random.split(subkey, n)
    # partial function to be able to send the agent as an argument
    # scan over the n steps
    (next_ep_state, next_ep_timestep), (cum_timestep) = (
        jax.lax.scan(step_fn, (state, timestep), random_keys)
    )
    return cum_timestep, next_ep_state, next_ep_timestep


def gather_data(state, timestep, subkey):
    keys = jax.random.split(subkey, params["num_batches"])
    timestep, next_ep_state, next_ep_timestep = jax.vmap(
        run_n_steps, in_axes=(0, 0, 0, None)
    )(state, timestep, keys, params["num_steps"])
    # print(timestep.reward.shape)
    # print(timestep.step_type.shape)

    return timestep, next_ep_state, next_ep_timestep

def step_fn(state_timestep, subkey):
    """A single step in the environment."""
    state, timestep = state_timestep

    # key = jax.random.PRNGKey(42)
    possible_actions = jnp.arange(4)
    random_action = jax.random.choice(subkey, possible_actions, shape=(1,), replace=True)[0]
    best_action = actions.action[0]

    state, timestep = env_step(state, best_action)

    return (state, timestep), (timestep)

env = jumanji.make("Game2048-v1")
wandb_logging.init_wandb(params)
env = AutoResetWrapper(env)
key = jax.random.PRNGKey(42)
rng_key, subkey = jax.random.split(key)
keys = jax.random.split(rng_key, params["num_batches"])

# Get the initial state and timestep
next_ep_state, next_ep_timestep = jax.vmap(env.reset)(keys)

prev_reward_arr = [[] for _ in range(params["num_batches"])]
prev_state_arr = [[next_ep_timestep.observation.board[i]] for i in range(params["num_batches"])]

for episode in range(1, 101):
    key, sample_key = jax.jit(jax.random.split)(key)
    timestep, next_ep_state, next_ep_timestep = gather_data(next_ep_state, next_ep_timestep, sample_key)

    prev_reward_arr, prev_state_arr = get_rewards(timestep, prev_reward_arr, prev_state_arr, episode)



