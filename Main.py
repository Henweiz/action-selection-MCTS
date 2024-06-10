import jax
import jax.numpy as jnp
from agents.agent import Agent
from agents.agent_2048 import Agent2048
from agents.agent_knapsack import AgentKnapsack 
from agents.agent_grid import AgentGrid
from agents.agent_maze import AgentMaze
import jumanji
from jumanji.wrappers import VmapAutoResetWrapper, AutoResetWrapper
from jumanji.types import StepType
import mctx
import functools 
from functools import partial
import flashbax as fbx

from plotting import plot_rewards, plot_losses

# Environments: Snake-v1, Knapsack-v1, Game2048-v1, Maze-v0
params = {
    "env_name": "Maze-v0",
    "agent": AgentMaze,
    "num_channels": 32, 
    "seed": 42,
    "lr": 0.001,
    "num_episodes": 50,
    "num_steps": 200,
    "num_actions": 4,
    "buffer_max_length": 50000,
    "buffer_min_length": 4,
    "num_batches": 32,
    "num_simulations": 16,
    "max_tree_depth": 4,
    "discount": 0.99,
}

class Timestep:
    def __init__(self, step_type, reward):
        self.step_type = step_type
        self.reward = reward


@jax.jit
def env_step(state, action):
    next_state, next_timestep = env.step(state, action)
    return next_state, next_timestep

def ep_loss_reward(timestep):
    # Apply the conditional reward change
    new_reward = jnp.where(timestep.step_type == StepType.LAST, -10, timestep.reward)
    return new_reward


def recurrent_fn(agent: Agent, rng_key, action, embedding):
    """One simulation step in MCTS."""
    del rng_key

    (state, timestep) = embedding
    new_state, new_timestep = env_step(state, action)
    prior_logits = agent.get_actions(new_timestep.observation)

    # TODO - transform back the value here
    value = agent.get_value(new_timestep.observation)



    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=timestep.reward,
        discount=params["discount"],
        prior_logits=prior_logits,
        value=value,
    )
    return recurrent_fn_output, (new_state, new_timestep)


def get_actions(agent, state, timestep, subkey):
    def root_fn(state, timestep, _):

        # TODO do transform with the value here?
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
    #print(actions.action)
    #print(actions.action_weights[0])
    #print(actions.reward)
    assert actions.action.shape[0] == 1
    assert actions.action_weights.shape[0] == 1
    best_action = jnp.argmax(actions.action_weights[0])
    state, timestep = env_step(state, best_action)
    q_value = actions.search_tree.summary().qvalues[
      0, best_action]
    #print(best_action)
    #print(q_value)
    return (state, timestep), (timestep, actions.action_weights[0], q_value)

def run_n_steps(state, timestep, subkey, agent, n):
    random_keys = jax.random.split(subkey, n)
    partial_step_fn = functools.partial(step_fn, agent)
    (state, timestep), (cum_timestep, actions, q_values) = jax.lax.scan(partial_step_fn, (state, timestep), random_keys)
    return cum_timestep, actions, q_values

def gather_data_new():
    key = jax.random.PRNGKey(params["seed"])
    rng_key, subkey = jax.random.split(key)

    keys = jax.random.split(rng_key, params["num_batches"])
    state, timestep = jax.vmap(env.reset)(keys)

    keys = jax.random.split(subkey, params["num_batches"])
    timestep, actions, q_values = jax.vmap(run_n_steps, in_axes=(0, 0, 0, None, None))(state, timestep, keys, agent, params["num_steps"])


    #
    # print(timestep)
    # print(actions)
    # print(q_values)

    return timestep, actions, q_values


def train(agent: Agent, rewards_arr, action_weights_arr, q_values_arr, states_arr):
    results_array = []

    for rewards, actions, q_values, states in zip(rewards_arr, action_weights_arr, q_values_arr, states_arr):

        # rewards = timestep.reward[batch_num]
        # TODO - transform the value here

        # steptypes = timestep.step_type[batch_num]
        # timesteps = Timestep(step_type=steptypes, reward=rewards)
        # rewards = ep_loss_reward(timesteps)

        results_array.append([
            jnp.sum(rewards).item(),
            jnp.max(rewards),
            agent.update_value(states, q_values),
            agent.update_policy(states, actions)
        ])

    avg_results_array = jnp.mean(jnp.array(results_array), axis=0)
    print(
        f"Total Return: {avg_results_array[0]} | Max Return: {avg_results_array[1]} | Value Loss: {str(round(avg_results_array[2], 6))} | Average Policy Loss: {str(round(avg_results_array[3], 6))}")

    return avg_results_array

@jax.jit
def process_episode(episode):
    print(f"Training Episode: {episode + 1}")
    timestep, actions, q_values = gather_data_new()
    avg_return = train(agent, timestep, actions, q_values)
    return avg_return


def get_dimension(env):
    if params["env_name"] == "Maze-v0":
        return [env.unwrapped.num_rows, env.unwrapped.num_cols]

if __name__ == "__main__":
    env = jumanji.make(params["env_name"])
    print(env)
    env = AutoResetWrapper(env)
    key = jax.random.PRNGKey(params["seed"])
    params["num_actions"] = env.action_spec.num_values
    agent = params.get("agent", Agent)(params, env)

    buffer = fbx.make_flat_buffer(max_length=params["buffer_max_length"], min_length=params["buffer_min_length"], sample_batch_size=params['num_batches'], add_batch_size=params['num_batches'])
    buffer = buffer.replace(
        init = jax.jit(buffer.init),
        add = jax.jit(buffer.add, donate_argnums=0),
        sample = jax.jit(buffer.sample),
        can_sample = jax.jit(buffer.can_sample),
    )

    dimensions = get_dimension(env)

    fake_timestep = {
        "q_value": jnp.zeros((params['num_steps'])),
        "actions": jnp.zeros((params['num_steps'], params['num_actions'])),
        "rewards": jnp.zeros((params['num_steps'])),
        "states": jnp.zeros((params['num_steps'], *dimensions, 1))
    }
    buffer_state = buffer.init(fake_timestep)

    all_results_array = []
    for episode in range(params["num_episodes"]):
        print(f"Training Episode: {episode + 1}")
        timestep, actions, q_values = gather_data_new()
        states = agent.get_state_from_observation(timestep.observation, True)

        buffer_state = buffer.add(buffer_state, {
            "q_value": q_values,
            "actions": actions,
            "rewards": timestep.reward,
            "states": states})

        if buffer.can_sample(buffer_state):
            key, sample_key = jax.jit(jax.random.split)(key)
            # does it make sense to sample the buffer more times?
            data = buffer.sample(buffer_state, sample_key).experience.first
            results_array = train(agent, data["rewards"], data["actions"], data["q_value"], data["states"])
            all_results_array.append(results_array)

    plot_rewards(all_results_array)
    plot_losses(all_results_array)


    key = jax.random.PRNGKey(params["seed"])
    state, timestep = env.reset(key)
    for step in range(params["num_steps"]):
        env.render(state)
        (new_state, new_timestep), (cum_timestep, actions, q_values) = step_fn(agent, (state, timestep), key)
        state = new_state
        timestep = new_timestep

