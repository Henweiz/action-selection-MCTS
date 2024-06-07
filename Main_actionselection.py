import jax
import jax.numpy as jnp
from agents.agent import Agent
from agents.agent_2048 import Agent2048
from agents.agent_knapsack import AgentKnapsack 
from agents.agent_grid import AgentGrid
import jumanji
from jumanji.wrappers import VmapAutoResetWrapper, AutoResetWrapper
from jumanji.types import StepType
import mctx
import functools 
from functools import partial
from action_selection_rules.solve_norm import ExPropKullbackLeibler, SquaredHellinger
from action_selection_rules.solve_trust_region import VariationalKullbackLeibler
from tree_policies import muzero_custom_policy

from plotting import plot_rewards, plot_losses

# Environments: Snake-v1, Knapsack-v1, Game2048-v1
params = {
    "env_name": "Snake-v1",
    "agent": AgentGrid,
    "policy": "KL_variational",
    "seed": 42,
    "lr": 0.001,
    "num_episodes": 100,
    "num_steps": 2000,
    "num_actions": 4,
    "buffer_max_length": 5000,
    "buffer_min_length": 1,
    "num_batches": 5,
    "num_simulations": 32,
    "max_tree_depth": 8,
    "num_channels": 32
}

policy_dict = {
    "default": mctx.muzero_policy,
    "KL_variational": functools.partial(muzero_custom_policy, selector=VariationalKullbackLeibler()),
    "KL_ex_prop": functools.partial(muzero_custom_policy, selector=ExPropKullbackLeibler()),
    "squared_hellinger": functools.partial(muzero_custom_policy, selector=SquaredHellinger()),
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


def recurrent_fn(params: Agent, rng_key, action, embedding):
    """One simulation step in MCTS."""
    del rng_key
    agent = params

    (state, timestep) = embedding
    new_state, new_timestep = env_step(state, action)
    prior_logits = agent.get_actions(new_timestep.observation)

    # TODO - transform back the value here
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

        # TODO do transform with the value here?
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
    state, timestep = state_timestep
    #print(timestep.observation.grid)
    actions = get_actions(agent, state, timestep, subkey)
    #print(actions)
    #print(actions.action_weights[1])
    #print(actions.reward)
    assert actions.action.shape[0] == 1
    assert actions.action_weights.shape[0] == 1
    best_action = actions.action[0]
    state, timestep = env_step(state, best_action)
    q_value = actions.search_tree.summary().qvalues[
      0, best_action]
    return (state, timestep), (timestep, actions.action_weights[1], q_value)

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

    return timestep, actions, q_values

def train(agent: Agent, timestep, action_weights, q_values):
    # total_reward_array, value_loss_array, policy_loss_array, max_reward_array = [], [], [], []
    results_array = []

    for batch_num, actions in enumerate(action_weights):
        states = agent.get_state_from_observation(timestep.observation, True)[batch_num]
        
        rewards = timestep.reward[batch_num]
        # TODO - transform the value here

        steptypes = timestep.step_type[batch_num]
        timesteps = Timestep(step_type=steptypes, reward=rewards)
        rewards = ep_loss_reward(timesteps)

        results_array.append([
            jnp.sum(rewards).item(),
            jnp.max(rewards),
            agent.update_value(states, q_values[batch_num]),
            agent.update_policy(states, actions)
        ])

    avg_results_array = jnp.mean(jnp.array(results_array), axis=0)
    print(f"Total Return: {avg_results_array[0]} | Max Return: {avg_results_array[1]} | Value Loss: {round(avg_results_array[2], 6)} | Average Policy Loss: {round(avg_results_array[3], 6)}")

    return avg_results_array

@jax.jit
def process_episode(episode):
    print(f"Training Episode: {episode + 1}")
    timestep, actions, q_values = gather_data_new()
    avg_return = train(agent, timestep, actions, q_values)
    return avg_return


if __name__ == "__main__":
    env = jumanji.make(params["env_name"])
    print(env)
    env = AutoResetWrapper(env)
    params["num_actions"] = env.action_spec.num_values
    agent = params.get("agent", Agent)(params, env)

    all_results_array = []
    for episode in range(params["num_episodes"]):
        print(f"Training Episode: {episode + 1}")
        timestep, actions, q_values = gather_data_new()
        results_array = train(agent, timestep, actions, q_values)
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