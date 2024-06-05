import jax
import jax.numpy as jnp
from agents.agent import Agent
from agents.agent_2048 import Agent2048
from agents.agent_knapsack import AgentKnapsack 
from agents.agent_grid import AgentGrid
import jumanji
from jumanji.wrappers import AutoResetWrapper
from jumanji.types import StepType
import mctx
import functools 
from functools import partial

# Environments: Snake-v1, Knapsack-v1, Game2048-v1
params = {
    "env_name": "Snake-v1",
    "agent": AgentGrid,
    "seed": 42,
    "lr": 0.01,
    "num_epochs": 10,
    "num_steps": 4000,
    "num_actions": 4,
    "buffer_max_length": 5000,
    "buffer_min_length": 1,
    "num_batches": 4,
    "num_simulations": 16,
    "max_tree_depth": 8
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

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=timestep.reward,
        discount=timestep.discount,
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
        num_simulations=params["num_simulations"],
        max_depth=params["max_tree_depth"],
        max_num_considered_actions=params["num_actions"],
        qtransform=partial(
            mctx.qtransform_completed_by_mix_value,
            value_scale=0.1,
            maxvisit_init=50,
            rescale_values=False,
        ),
        gumbel_scale=1.0,
    )
    return policy_output


def step_fn(agent, state_timestep, subkey):
    state, timestep = state_timestep
    #print(timestep.observation.grid)
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

        print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}, Returns: {jnp.sum(returns)}")

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
    key = jax.random.PRNGKey(params["seed"])
    state, timestep = env.reset(key)
    for step in range(params["num_steps"]):
        env.render(state)
        (new_state, new_timestep), (cum_timestep, actions) = step_fn(agent, (state, timestep), key)
        state = new_state
        timestep = new_timestep