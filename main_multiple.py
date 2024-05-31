import jax
import jax.numpy as jnp
import flashbax as fbx
from flax.training import train_state
from flax import linen as nn
import optax
from agent import AlphaZero
import jumanji
from jumanji.wrappers import AutoResetWrapper
from jumanji.types import StepType
import mctx
import functools

params = {
    "env_name": "Game2048-v1",
    "seed": 42,
    "lr": 0.01,
    "num_epochs": 3,
    "num_steps": 2000,
    "num_actions": 4,
    "buffer_max_length": 5000,
    "buffer_min_length": 1,
    "num_batches": 2,
}


@jax.jit
def env_step(timestep, action):
    next_state, next_timestep = env.step(timestep, action)
    return next_state, next_timestep


def recurrent_fn(params: AlphaZero, rng_key, action, embedding):
    """One simulation step in MCTS."""
    del rng_key
    agent = params

    new_embedding, timestep = env_step(embedding, action)
    prior_logits = agent.get_actions(new_embedding)
    value = agent.get_value(new_embedding)
    discount = timestep.discount

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=timestep.reward,
        discount=discount,
        prior_logits=prior_logits,
        value=value,
    )
    return recurrent_fn_output, new_embedding


def get_actions(agent, state, subkey):
    def root_fn(state, _):
        root = mctx.RootFnOutput(
            prior_logits=agent.get_actions(state),
            value=agent.get_value(state),
            embedding=state,
        )
        return root

    policy_output = mctx.gumbel_muzero_policy(
        params=agent,
        rng_key=subkey,
        root=jax.vmap(root_fn, (None, 0))(state, jnp.ones(1)),  # params["num_steps"])),
        recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
        num_simulations=8,
        max_depth=4,
        max_num_considered_actions=params["num_actions"],
    )
    return policy_output


@jax.jit
def step_fn(agent, state, subkey):
    actions = get_actions(agent, state, subkey)
    best_action = actions.action[0]
    state, timestep = env_step(state, best_action)
    return state, (timestep, actions.action_weights[0])

@jax.jit
def run_n_steps(state, subkey, agent, n):
    random_keys = jax.random.split(subkey, n)
    partial_step_fn = functools.partial(step_fn, agent)
    state, (timestep, actions) = jax.lax.scan(partial_step_fn, state, random_keys)

    return timestep, actions

def gather_data_new():
    key = jax.random.PRNGKey(params["seed"])
    rng_key, subkey = jax.random.split(key)

    keys = jax.random.split(rng_key, params["num_batches"])
    state, timestep = jax.vmap(env.reset)(keys)

    keys = jax.random.split(subkey, params["num_batches"])
    timestep, actions = jax.vmap(run_n_steps, in_axes=(0, 0, None, None))(state, keys, agent, params["num_steps"])

    return timestep, actions

def train(agent, timestep, action_weights):
    for batch_num, actions in enumerate(action_weights):
        states = timestep.observation.board[batch_num]
        returns = timestep.reward[batch_num]

        value_loss = agent.update_value(states, returns)
        policy_loss = agent.update_policy(states, actions)

        print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}, Returns: {jnp.sum(returns)}, Max returns: {jnp.max(returns)}")


if __name__ == "__main__":
    env = jumanji.make(params["env_name"])
    env = AutoResetWrapper(env)
    params["num_actions"] = env.action_spec.num_values
    print(f"Action Spec: {env.action_spec}")
    agent = AlphaZero(params, env)

    for epoch in range(params["num_epochs"]):
        print(f"Training Epoch: {epoch + 1}")
        timestep, actions = gather_data_new()
        train(agent, timestep, actions)





