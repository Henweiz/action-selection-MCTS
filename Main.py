import jax
import jax.numpy as jnp
from agents.agent import Agent
from agents.agent_2048 import Agent2048
from agents.agent_knapsack import AgentKnapsack
import jumanji
from jumanji.wrappers import AutoResetWrapper
from jumanji.types import StepType
import mctx
import functools

params = {
    "env_name": "Knapsack-v1",
    "agent": AgentKnapsack,
    "seed": 42,
    "lr": 0.01,
    "num_epochs": 5,
    "num_steps": 50,
    "num_actions": 50,
    "buffer_max_length": 5000,
    "buffer_min_length": 1,
    "num_batches": 5,
}


@jax.jit
def env_step(state, action):
    next_state, next_timestep = env.step(state, action)
    return next_state, next_timestep


def recurrent_fn(params: Agent, rng_key, action, embedding):
    """One simulation step in MCTS."""
    del rng_key
    agent = params

    (state, timestep) = embedding
    new_state, new_timestep = env_step(state, action)
    prior_logits = agent.get_actions(new_timestep.observation)
    value = agent.get_value(new_timestep.observation)
    discount = timestep.discount

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=timestep.reward,
        discount=discount,
        prior_logits=prior_logits,
        value=value,
    )
    return recurrent_fn_output, (new_state, new_timestep)


def get_actions(agent, state, timestep, subkey):
    def root_fn(state, timestep, _):
        root = mctx.RootFnOutput(
            prior_logits=agent.get_actions(timestep.observation),
            value=agent.get_value(timestep.observation),
            embedding=(state, timestep),
        )
        return root

    policy_output = mctx.gumbel_muzero_policy(
        params=agent,
        rng_key=subkey,
        root=jax.vmap(root_fn, (None, None, 0))(state, timestep, jnp.ones(1)),  # params["num_steps"])),
        recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
        num_simulations=50,
        max_depth=10,
        max_num_considered_actions=params["num_actions"],
    )
    return policy_output


def step_fn(agent, state_timestep, subkey):
    state, timestep = state_timestep
    actions = get_actions(agent, state, timestep, subkey)
    assert actions.action.shape[0] == 1
    assert actions.action_weights.shape[0] == 1
    best_action = actions.action[0]
    state, timestep = env_step(state, best_action)
    return (state, timestep), (timestep, actions.action_weights[0])

def run_n_steps(state, timestep, subkey, agent, n):
    random_keys = jax.random.split(subkey, n)
    partial_step_fn = functools.partial(step_fn, agent)
    (state, timestep), (cum_timestep, actions) = jax.lax.scan(partial_step_fn, (state, timestep), random_keys)
    return cum_timestep, actions

def gather_data_new():
    key = jax.random.PRNGKey(params["seed"])
    rng_key, subkey = jax.random.split(key)

    keys = jax.random.split(rng_key, params["num_batches"])
    state, timestep = jax.vmap(env.reset)(keys)

    keys = jax.random.split(subkey, params["num_batches"])
    timestep, actions = jax.vmap(run_n_steps, in_axes=(0, 0, 0, None, None))(state, timestep, keys, agent, params["num_steps"])

    return timestep, actions

def train(agent: Agent, timestep, action_weights):
    for batch_num, actions in enumerate(action_weights):
        states = agent.get_state_from_observation(timestep.observation, True)[batch_num]
        returns = timestep.reward[batch_num]

        value_loss = agent.update_value(states, returns)
        policy_loss = agent.update_policy(states, actions)

        print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}, Returns: {jnp.sum(returns)}, Max returns: {jnp.max(returns)}")

if __name__ == "__main__":
    env = jumanji.make(params["env_name"])
    print(env)
    env = AutoResetWrapper(env)
    params["num_actions"] = env.action_spec.num_values
    agent = params.get("agent", Agent)(params, env)

    for epoch in range(params["num_epochs"]):
        print(f"Training Epoch: {epoch + 1}")
        timestep, actions = gather_data_new()
        train(agent, timestep, actions)