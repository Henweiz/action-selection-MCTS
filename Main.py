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
import matplotlib.pyplot as plt

# Environments: Snake-v1, Knapsack-v1, Game2048-v1, Maze-v0
params = {
    "env_name": "Maze-v0",
    "agent": AgentMaze,
    "num_channels": 32, 
    "seed": 42,
    "lr": 0.01,
    "discount": 0.99,
    "num_episodes": 200,
    "num_steps": 100,
    "num_actions": 4,
    "buffer_max_length": 5000,
    "buffer_min_length": 1,
    "num_batches": 1,
    "num_simulations": 32,
    "max_tree_depth": 8
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

def gather_data_new(key):
    #key = jax.random.PRNGKey(params["seed"])
    rng_key, subkey = jax.random.split(key)

    keys = jax.random.split(rng_key, params["num_batches"])
    state, timestep = jax.vmap(env.reset)(keys)

    keys = jax.random.split(subkey, params["num_batches"])
    timestep, actions, q_values = jax.vmap(run_n_steps, in_axes=(0, 0, 0, None, None))(state, timestep, keys, agent, params["num_steps"])

    return timestep, actions, q_values

def train(agent: Agent, timestep, action_weights, q_values):
    avg_return = []
    for batch_num, actions in enumerate(action_weights):
        states = agent.get_state_from_observation(timestep.observation, True)[batch_num]
        
        rewards = timestep.reward[batch_num]
        #steptypes = timestep.step_type[batch_num]
        #timesteps = Timestep(step_type=steptypes, reward=rewards)
        #rewards = ep_loss_reward(timesteps)

        value_loss = agent.update_value(states, q_values[batch_num])
        policy_loss = agent.update_policy(states, actions)
        returns = jnp.sum(rewards)
        #print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}, Returns: {returns}")
        avg_return.append(returns)
    #avg_return_array = jnp.array(avg_return)   
    #avg_return_array =  jnp.mean(avg_return_array)
    #print(f"Avg return: {avg_return_array}")
    return avg_return

#@jax.jit
def process_episode(key):
    timestep, actions, q_values = gather_data_new(key)
    #print(q_values)
    avg_return = train(agent, timestep, actions, q_values)
    return avg_return

if __name__ == "__main__":
    env = jumanji.make(params["env_name"])
    print(env)
    env = AutoResetWrapper(env)
    key = jax.random.PRNGKey(params["seed"])
    params["num_actions"] = env.action_spec.num_values
    agent = params.get("agent", Agent)(params, env)
    avg_return = []
    keys = jax.random.split(key, params["num_episodes"])


    for episode in range(params["num_episodes"]):
        print(f"Training episode: {episode + 1}")
        avg_returns = process_episode(keys[episode])
        avg_returns = jnp.array(avg_returns)   
        avg_returns = jnp.mean(avg_returns)
        print(avg_returns)
        avg_return.append(avg_returns)

    avg_return = jnp.array(avg_return)   
    print(avg_return)
    # Create a range of episodes
    episodes = jnp.arange(len(avg_return))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, avg_return, marker='o', linestyle='-')
    plt.title('Returns per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(True)
    plt.show()

    state, timestep = env.reset(key)
    for step in range(params["num_steps"]):
        env.render(state)
        (new_state, new_timestep), (cum_timestep, actions, q_values) = step_fn(agent, (state, timestep), key)
        state = new_state
        timestep = new_timestep

