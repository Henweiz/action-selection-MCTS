import jax
import jax.numpy as jnp
from jax import random

from agents.agent import Agent
from agents.agent_2048 import Agent2048

from agents.agent_maze import AgentMaze
import jumanji
from jumanji.wrappers import VmapAutoResetWrapper, AutoResetWrapper
from jumanji.types import StepType
import mctx
import functools 
from functools import partial
import flashbax as fbx

from action_selection_rules.solve_norm import ExPropKullbackLeibler, SquaredHellinger
from action_selection_rules.solve_trust_region import VariationalKullbackLeibler
from tree_policies import muzero_custom_policy



from plotting import plot_rewards, plot_losses

# Environments: Snake-v1, Knapsack-v1, Game2048-v1, Maze-v0
params = {
    "env_name": "Maze-v0",
    "policy": "default",
    "agent": AgentMaze,
    "num_channels": 32, 
    "seed": 43,
    "lr": 0.01, # 0.00003
    "num_episodes": 200,
    "num_steps": 50,
    "num_actions": 4,
    "buffer_max_length": 50000,
    "buffer_min_length": 4,
    "num_batches": 64,
    "num_simulations": 16,
    "max_tree_depth": 12,
    "discount": 0.99,
}

policy_dict = {
    "default": mctx.muzero_policy,
    "KL_variational": functools.partial(muzero_custom_policy, selector=VariationalKullbackLeibler()),
    "KL_ex_prop": functools.partial(muzero_custom_policy, selector=ExPropKullbackLeibler()),
    "squared_hellinger": functools.partial(muzero_custom_policy, selector=SquaredHellinger()),
}


class Timestep:
    """Tuple for storing the step type and reward together.
    TODO Consider renaming to avoid confusion with the environment timestep.

    Attributes:
        step_type: The type of the step (e.g., LAST).
        reward: The reward received at this timestep.
    """

    def __init__(self, step_type, reward):
        self.step_type = step_type
        self.reward = reward



@jax.jit
def env_step(state, action):
    """A single step in the environment."""
    next_state, next_timestep = env.step(state, action)
    return next_state, next_timestep

def ep_loss_reward(timestep):
    """Reward transformation for the environment."""
    new_reward = jnp.where(timestep.step_type == StepType.LAST, -10, timestep.reward)
    return new_reward


def recurrent_fn(agent: Agent, rng_key, action, embedding):
    """One simulation step in MCTS."""
    del rng_key

    (state, timestep) = embedding
    new_state, new_timestep = env_step(state, action)

    # get the action probabilities from the network
    prior_logits = agent.get_actions(new_timestep.observation)

    # get the value from the network
    value = agent.get_value(new_timestep.observation)

    # return the recurrent function output
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=timestep.reward,
        discount=params["discount"],
        prior_logits=prior_logits,
        value=value,
    )
    return recurrent_fn_output, (new_state, new_timestep)


def get_actions(agent, state, timestep, subkey):
    """Get the actions from the MCTS"""

    def root_fn(state, timestep, _):
        """Root function for the MCTS."""

        root = mctx.RootFnOutput(
            prior_logits=agent.get_actions(timestep.observation),
            value=agent.get_value(timestep.observation),
            embedding=(state, timestep),
        )
        return root

    policy = policy_dict[params["policy"]]

    policy_output = policy(
        params=agent,
        rng_key=subkey,
        root=jax.vmap(root_fn, (None, None, 0))(state, timestep, jnp.ones(1)),  # params["num_steps"])),
        recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
        num_simulations=params["num_simulations"],
        max_depth=params["max_tree_depth"],
        # max_num_considered_actions=params["num_actions"],
        qtransform=partial(
            mctx.qtransform_completed_by_mix_value,
            value_scale=0.1,
            maxvisit_init=50,
            rescale_values=False,
        ),
        # gumbel_scale=1.0,
    )
    return policy_output


def step_fn(agent, state_timestep, subkey):
    """A single step in the environment."""
    state, timestep = state_timestep
    actions = get_actions(agent, state, timestep, subkey)

    assert actions.action.shape[0] == 1
    assert actions.action_weights.shape[0] == 1

    # key = jax.random.PRNGKey(42)
    best_action = actions.action[0]

    state, timestep = env_step(state, best_action)
    q_value = actions.search_tree.summary().qvalues[
      actions.search_tree.ROOT_INDEX, best_action]
    #weights = jax.nn.one_hot(best_action, params["num_actions"])
    #print(best_action)
    #print(q_value)
    return (state, timestep), (timestep, actions.action_weights[0], q_value)

def run_n_steps(state, timestep, subkey, agent, n):
    random_keys = jax.random.split(subkey, n)
    partial_step_fn = functools.partial(step_fn, agent)
    (next_ep_state, next_ep_timestep), (cum_timestep, actions, q_values) = jax.lax.scan(partial_step_fn, (state, timestep), random_keys)
    return cum_timestep, actions, q_values, next_ep_state, next_ep_timestep

def gather_data_new(state, timestep, subkey):
    keys = jax.random.split(subkey, params["num_batches"])
    timestep, actions, q_values, next_ep_state, next_ep_timestep = jax.vmap(run_n_steps, in_axes=(0, 0, 0, None, None))(state, timestep, keys, agent, params["num_steps"])

    return timestep, actions, q_values, next_ep_state, next_ep_timestep


def train(agent: Agent, rewards_arr, action_weights_arr, q_values_arr, states_arr):
    results_array = []

    for rewards, actions, q_values, states in zip(rewards_arr, action_weights_arr, q_values_arr, states_arr):
        # TODO - transform the value here?

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
        f"Value Loss: {str(round(avg_results_array[2], 6))} | Policy Loss: {str(round(avg_results_array[3], 6))}")

    return avg_results_array


def get_dimension(env):
    if params["env_name"] == "Maze-v0":
        return [env.unwrapped.num_rows, env.unwrapped.num_cols]

if __name__ == "__main__":
    env = jumanji.make(params["env_name"])
    print(f"running {params['env_name']}")
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

    # Initialize the buffer
    fake_timestep = {
        "q_value": jnp.zeros((params['num_steps'])),
        "actions": jnp.zeros((params['num_steps'], params['num_actions'])),
        "rewards": jnp.zeros((params['num_steps'])),
        "states": jnp.zeros((params['num_steps'], *agent.input_shape))
    }
    buffer_state = buffer.init(fake_timestep)

    all_results_array = []
    avg_rewards = []

    key = jax.random.PRNGKey(params["seed"])
    rng_key, subkey = jax.random.split(key)

    keys = jax.random.split(rng_key, params["num_batches"])
    next_ep_state, next_ep_timestep = jax.vmap(env.reset)(keys)

    for episode in range(params["num_episodes"]):
        timestep, actions, q_values, next_ep_state, next_ep_timestep = gather_data_new(next_ep_state, next_ep_timestep, subkey)

        states = agent.get_state_from_observation(timestep.observation, True)
        avg = jnp.sum(timestep.reward).item() / params["num_batches"]
        print(f"Episode {episode + 1} avg reward: {avg}")
        avg_rewards.append(avg)

        buffer_state = buffer.add(buffer_state, {
            "q_value": q_values,
            "actions": actions,
            "rewards": timestep.reward,
            "states": states})

        if buffer.can_sample(buffer_state):
            key, sample_key = jax.jit(jax.random.split)(key)
            # does it make sense to sample the buffer more times?
            data = buffer.sample(buffer_state, sample_key).experience.first
            #next_data = buffer.sample(buffer_state, sample_key).experience.second
            #print(data)
            results_array = train(agent, data["rewards"], data["actions"], data["q_value"], data["states"])
            results_array = results_array.at[0].set(avg)
            all_results_array.append(results_array)

    plot_rewards(all_results_array)
    plot_losses(all_results_array)
    print(avg_rewards)


    key = jax.random.PRNGKey(params["seed"])
    state, timestep = env.reset(key)
    for step in range(params["num_steps"]):
        env.render(state)
        (new_state, new_timestep), (cum_timestep, actions, q_values) = step_fn(agent, (state, timestep), key)
        state = new_state
        timestep = new_timestep

