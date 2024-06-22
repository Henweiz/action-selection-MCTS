import functools
from functools import partial
from typing import Optional

import flashbax as fbx
import jax
import jax.numpy as jnp
import jumanji
import mctx
from flashbax.vault import Vault
from jumanji.environments.routing.maze import generator
from jumanji.types import StepType
from jumanji.wrappers import AutoResetWrapper

import wandb
from action_selection_rules.solve_norm import ExPropKullbackLeibler, SquaredHellinger
from action_selection_rules.solve_trust_region import VariationalKullbackLeibler
from agents.agent import Agent
from agents.agent_knapsack import AgentKnapsack
from tree_policies import muzero_custom_policy
from wandb_logging import init_wandb

# Environments: Snake-v1, Knapsack-v1, Game2048-v1, Maze-v0
params = {
    "env_name": "Knapsack-v1",
    "maze_size": (5, 5),
    "policy": "default",
    "agent": AgentKnapsack,
    "num_channels": 32,
    "seed": 42,
    "lr": 2e-4,
    "num_episodes": 40,
    "num_steps": 100,
    "num_actions": 4,
    "obs_spec": Optional,
    "buffer_max_length": 2000,
    "buffer_min_length": 2,
    "num_batches": 8,
    "sample_size": 64,
    "num_simulations": 32,
    "max_tree_depth": 12,
    "discount": 1,
    "logging": True,
    "run_in_kaggle": False,
    "checkpoint_dir": r"/home/iwitko/repos/action-selection-MCTS/checkpoints",
    "checkpoint_interval": 5,
    "load_checkpoint": False,
}

policy_dict = {
    "default": mctx.muzero_policy,
    "KL_variational": functools.partial(
        muzero_custom_policy, selector=VariationalKullbackLeibler()
    ),
    "KL_ex_prop": functools.partial(
        muzero_custom_policy, selector=ExPropKullbackLeibler()
    ),
    "squared_hellinger": functools.partial(
        muzero_custom_policy, selector=SquaredHellinger()
    ),
}
class Steptype_Reward_Tuple:
    """Tuple for storing the step type and reward together.

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


def recurrent_fn(agent: Agent, _, action, embedding):
    """One simulation step in MCTS."""

    (state, timestep) = embedding

    # take a step in the environment
    new_state, new_timestep = env_step(state, action)

    # get the action probabilities from the network
    prior_logits, value = agent.get_output(new_timestep.observation)

    # return the recurrent function output
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=timestep.reward,
        discount=params["discount"],
        prior_logits=prior_logits,
        value=value,
    )
    return recurrent_fn_output, (new_state, new_timestep)


def get_actions(agent, state, timestep, subkey) -> mctx.PolicyOutput:
    """Get the actions from the MCTS"""

    def root_fn(state, timestep, _):
        """TODO can someone explain what this function does?"""
        priors, value = agent.get_output(timestep.observation)

        root = mctx.RootFnOutput(
            prior_logits=priors,
            value=value,
            embedding=(state, timestep),
        )
        return root

    # get the selected action selection policy
    policy = policy_dict[params["policy"]]

    policy_output = policy(
        params=agent,
        rng_key=subkey,
        root=jax.vmap(root_fn, (None, None, 0))(
            state, timestep, jnp.ones(1)
        ),
        recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
        num_simulations=params["num_simulations"],
        max_depth=params["max_tree_depth"],
        qtransform=partial(
            mctx.qtransform_completed_by_mix_value,
            value_scale=0.1,
            maxvisit_init=50,
            rescale_values=False,
        ),
    )

    return policy_output


def get_rewards(timestep, prev_reward_arr, episode):
    """Get the rewards from the timestep and log them to wandb."""
    rewards = []
    new_reward_arr = []
    max_reward = jnp.max(timestep.reward)

    for batch_num in range(len(prev_reward_arr)):

        # the previous total rewards for all games
        prev_reward = prev_reward_arr[batch_num]

        # go over all steps in the batch
        for i, (step_type, ep_rew) in enumerate(
                zip(timestep.step_type[batch_num], timestep.reward[batch_num])
        ):
            # if the episode has ended, add the total reward to the rewards list
            if step_type == StepType.LAST:
                # the reward of this game and the max reward of the last num_steps steps
                rew = {
                    "reward": prev_reward + ep_rew,
                    "max_reward": max_reward,
                }
                rewards.append(rew)

                # reset the previous total reward
                prev_reward = 0

                # log the reward to wandb
                if params["logging"]:
                    wandb.log(
                        rew,
                        step=(episode - 1) * params["num_batches"] * params["num_steps"]
                             + batch_num * params["num_steps"]
                             + (i + 1),
                    )
            else:
                # add the reward to the previous total reward if the game has not ended
                prev_reward += ep_rew

        # add to the new array containing the total rewards of all games so far
        new_reward_arr.append(prev_reward)

    # calculate the average reward of all games that has finished in the last num_steps steps
    avg_reward = sum([r["reward"] for r in rewards]) / max(1, len(rewards))

    steps = (episode - 1) * params["num_batches"] * params["num_steps"] + params["num_steps"]

    print(
        f"Episode {episode}, Average reward: {str(round(avg_reward, 1))}, Max Reward: {max_reward},"
        f" Steps: {steps} / {params['num_episodes'] * params['num_batches'] * params['num_steps']}"
    )

    return new_reward_arr



def step_fn(agent, state_timestep, subkey):
    """A single step in the environment."""
    state, timestep = state_timestep
    actions = get_actions(agent, state, timestep, subkey)

    assert actions.action.shape[0] == 1
    assert actions.action_weights.shape[0] == 1

    # The action suggested by the MCTS
    best_action = actions.action[0]

    # Take a step in the environment
    state, timestep = env_step(state, best_action)

    # TODO explain this
    q_value = actions.search_tree.summary().qvalues[
        actions.search_tree.ROOT_INDEX, best_action
    ]

    return (state, timestep), (
        timestep,
        actions.action_weights[0],
        q_value,
        actions.search_tree,
    )


def run_n_steps(state, timestep, subkey, agent, n):
    """Run n steps in the environment in parallel"""

    random_keys = jax.random.split(subkey, n)

    # partial function to be able to send the agent as an argument
    partial_step_fn = functools.partial(step_fn, agent)

    # run n steps in the environment in parallel
    (next_ep_state, next_ep_timestep), (cum_timestep, actions, q_values, trees) = (
        jax.lax.scan(partial_step_fn, (state, timestep), random_keys)
    )
    return cum_timestep, actions, q_values, next_ep_state, next_ep_timestep, trees


def gather_data(state, timestep, subkey):
    """Gather data by playing num_steps steps in the environment."""
    keys = jax.random.split(subkey, params["num_batches"])

    timestep, actions, q_values, next_ep_state, next_ep_timestep, trees = jax.vmap(
        run_n_steps, in_axes=(0, 0, 0, None, None)
    )(state, timestep, keys, agent, params["num_steps"])

    return timestep, actions, q_values, next_ep_state, next_ep_timestep, trees


def train(agent: Agent, action_weights_arr, q_values_arr, states_arr, episode):
    """Train the agent on the gathered data."""
    losses = [
        agent.update_fn(states, actions, q_values, episode)
        for actions, q_values, states in zip(
            action_weights_arr, q_values_arr, states_arr
        )
    ]

    return jnp.mean(jnp.array(losses), axis=0)

def get_checkpoint_id():
    return f"{params['env_name']}_{params['policy']}_{params['num_simulations']}_{params['seed']}"


if __name__ == "__main__":

    # Initialize wandb
    if params["logging"]:
        init_wandb(params)

    # Initialize the environment
    if params["env_name"] == "Maze-v0":
        gen = generator.RandomGenerator(*params["maze_size"])
        env = jumanji.make(params["env_name"], generator=gen)
    else:
        env = jumanji.make(params["env_name"])

    print(f"running {params['env_name']}")
    env = AutoResetWrapper(env)

    # Initialize the agent
    params["num_actions"] = env.action_spec.num_values
    params["obs_spec"] = env.observation_spec
    agent = params.get("agent", Agent)(params)

    vault = None
    buffer = None

    # If we are starting a run again from a checkpoint
    if params["load_checkpoint"]:
        vault = Vault(
            "buffer_checkpoint",
            None,
            vault_uid=get_checkpoint_id(),
            rel_dir="checkpoints",
        )
        buffer = vault.read(params["buffer_max_length"])
        agent.load(params["checkpoint_dir"], get_checkpoint_id())
        # TODO add the network loading here?
    else:
        # Specify buffer parameters
        buffer = fbx.make_flat_buffer(
            max_length=params["buffer_max_length"],
            min_length=params["buffer_min_length"],
            sample_batch_size=params["sample_size"],
            add_batch_size=params["num_batches"],
        )

        # Jit the buffer functions
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        # Specify buffer format
        states_as_ints = params["env_name"] == "Game2048-v1"

        fake_timestep = {
            "q_value": jnp.zeros((params["num_steps"])),
            "actions": jnp.zeros((params["num_steps"], params["num_actions"]), dtype=jnp.float32),
            "states": jnp.zeros((params["num_steps"], *agent.input_shape), dtype=jnp.int32 if states_as_ints else jnp.float32),
        }
        buffer_state = buffer.init(fake_timestep)
        vault = Vault(
            "buffer_checkpoint", buffer_state.experience, vault_uid=get_checkpoint_id(), rel_dir="checkpoints"
        )

    # Initialize the random keys
    key = jax.random.PRNGKey(params["seed"])
    rng_key, subkey = jax.random.split(key)
    keys = jax.random.split(rng_key, params["num_batches"])

    # Get the initial state and timestep
    next_ep_state, next_ep_timestep = jax.vmap(env.reset)(keys)

    # The previous rewards for the game in each batch
    prev_reward_arr = [0 for _ in range(params["num_batches"])]

    for episode in range(1, params["num_episodes"] + 1):

        # Get new key every episode
        key, sample_key = jax.jit(jax.random.split)(key)

        # Gather data from the environment
        timestep, actions, q_values, next_ep_state, next_ep_timestep, trees = (
            gather_data(next_ep_state, next_ep_timestep, sample_key)
        )

        # Save the previous total rewards for the game in each batch
        prev_reward_arr = get_rewards(timestep, prev_reward_arr, episode)

        # Get state in the correct format given environment
        states = agent.get_state_from_observation(timestep.observation, True)

        # Add data to buffer
        buffer_state = buffer.add(
            buffer_state,
            {
                "q_value": q_values,
                "actions": actions,
                "states": states,
            },
        )
        vault.write(buffer_state)

        if buffer.can_sample(buffer_state):
            # Create a new key every time
            key, sample_key = jax.jit(jax.random.split)(key)

            # Get the data to train from the buffer
            data = buffer.sample(buffer_state, sample_key).experience.first
            loss = train(
                agent, data["actions"], data["q_value"], data["states"], episode
            )
            agent.log_losses(episode, params)

        # Save the network every checkpoint_interval episodes
        if episode % params["checkpoint_interval"] == 0:
            print(f"Saving checkpoint for episode {episode}")
            agent.save(params["checkpoint_dir"], episode, get_checkpoint_id())

    if params["logging"]:
        wandb.finish()
