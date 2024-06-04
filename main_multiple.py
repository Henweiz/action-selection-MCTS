# import jax
# import jax.numpy as jnp
# import flashbax as fbx
# from flax.training import train_state
# from flax import linen as nn
# import optax
# from agent import AlphaZero
# import jumanji
# from jumanji.wrappers import AutoResetWrapper
# from jumanji.types import StepType
# import mctx
# import functools
#
# params = {
#     "env_name": "Game2048-v1",
#     "seed": 42,
#     "lr": 0.01,
#     "num_epochs": 3,
#     "num_steps": 2000,
#     "num_actions": 4,
#     "buffer_max_length": 5000,
#     "buffer_min_length": 1,
#     "num_batches": 2,
# }
#
#
# @jax.jit
# def env_step(timestep, action):
#     next_state, next_timestep = env.step(timestep, action)
#     return next_state, next_timestep
#
#
# def recurrent_fn(params: AlphaZero, rng_key, action, embedding):
#     """One simulation step in MCTS."""
#     del rng_key
#     agent = params
#
#     new_embedding, timestep = env_step(embedding, action)
#     prior_logits = agent.get_actions(new_embedding)
#     value = agent.get_value(new_embedding)
#     discount = timestep.discount
#
#     recurrent_fn_output = mctx.RecurrentFnOutput(
#         reward=timestep.reward,
#         discount=discount,
#         prior_logits=prior_logits,
#         value=value,
#     )
#     return recurrent_fn_output, new_embedding
#
#
# def get_actions(agent, state, subkey):
#     def root_fn(state, _):
#         root = mctx.RootFnOutput(
#             prior_logits=agent.get_actions(state),
#             value=agent.get_value(state),
#             embedding=state,
#         )
#         return root
#
#     policy_output = mctx.gumbel_muzero_policy(
#         params=agent,
#         rng_key=subkey,
#         root=jax.vmap(root_fn, (None, 0))(state, jnp.ones(1)),  # params["num_steps"])),
#         recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
#         num_simulations=8,
#         max_depth=4,
#         max_num_considered_actions=params["num_actions"],
#     )
#     return policy_output
#
#
# @jax.jit
# def step_fn(agent, state, subkey):
#     actions = get_actions(agent, state, subkey)
#     best_action = actions.action[0]
#     state, timestep = env_step(state, best_action)
#     return state, (timestep, actions.action_weights[0])
#
# @jax.jit
# def run_n_steps(state, subkey, agent, n):
#     random_keys = jax.random.split(subkey, n)
#     partial_step_fn = functools.partial(step_fn, agent)
#     state, (timestep, actions) = jax.lax.scan(partial_step_fn, state, random_keys)
#
#     return timestep, actions
#
# def gather_data_new():
#     key = jax.random.PRNGKey(params["seed"])
#     rng_key, subkey = jax.random.split(key)
#
#     keys = jax.random.split(rng_key, params["num_batches"])
#     state, timestep = jax.vmap(env.reset)(keys)
#
#     keys = jax.random.split(subkey, params["num_batches"])
#     timestep, actions = jax.vmap(run_n_steps, in_axes=(0, 0, None, None))(state, keys, agent, params["num_steps"])
#
#     return timestep, actions
#
# def train(agent, timestep, action_weights):
#     for batch_num, actions in enumerate(action_weights):
#         states = timestep.observation.board[batch_num]
#         returns = timestep.reward[batch_num]
#
#         value_loss = agent.update_value(states, returns)
#         policy_loss = agent.update_policy(states, actions)
#
#         print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}, Returns: {jnp.sum(returns)}, Max returns: {jnp.max(returns)}")
#
#
# if __name__ == "__main__":
#     env = jumanji.make(params["env_name"])
#     env = AutoResetWrapper(env)
#     params["num_actions"] = env.action_spec.num_values
#     print(f"Action Spec: {env.action_spec}")
#     agent = AlphaZero(params, env)
#
#     for epoch in range(params["num_epochs"]):
#         print(f"Training Epoch: {epoch + 1}")
#         timestep, actions = gather_data_new()
#         train(agent, timestep, actions)
#

import jax
import jax.numpy as jnp
import flashbax as fbx
from flax.training import train_state
from flax import linen as nn
import optax
from agents.agent_2048 import Agent2048
import jumanji
from jumanji.wrappers import AutoResetWrapper
from jumanji.types import StepType
import mctx
import numpy as np
import matplotlib.pyplot as plt

params = {
    "env_name": "Game2048-v1",
    "agent": Agent2048,
    "seed": 42,
    "lr": 0.01,
    "num_epochs": 1,
    "num_steps": 5,
    "num_actions": 4,
    "buffer_max_length": 5000,
    "buffer_min_length": 1,
    "num_batches": 2,
    "num_trials": 10
}

@jax.jit
def env_step(timestep, action):
    next_state, next_timestep = env.step(timestep, action)
    return next_state, next_timestep

def recurrent_fn(params: AlphaZero, rng_key, action, embedding):
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

def select_action(policy_output, selection_rule="greedy", epsilon=0.1, temperature=1.0, c=1.0):
    action_weights = policy_output.action_weights[0]
    if selection_rule == "greedy":
        best_action = policy_output.action[0]
    elif selection_rule == "epsilon_greedy":
        if np.random.rand() < epsilon:
            best_action = np.random.choice(len(action_weights))
        else:
            best_action = policy_output.action[0]
    elif selection_rule == "softmax":
        action_probabilities = jax.nn.softmax(action_weights / temperature)
        action_probabilities = np.array(action_probabilities)  # Convert to numpy array
        action_probabilities /= np.sum(action_probabilities)  # Ensure the probabilities sum to 1
        best_action = np.random.choice(len(action_weights), p=action_probabilities)
    elif selection_rule == "ucb":
        total_visits = np.sum(policy_output.visit_counts)
        ucb_values = action_weights + c * np.sqrt(2 * np.log(total_visits) / (policy_output.visit_counts + 1))
        best_action = np.argmax(ucb_values)
    else:
        raise ValueError("Unsupported selection rule")
    return best_action

def get_actions(agent, state, subkey, selection_rule="greedy", epsilon=0.1, temperature=1.0, c=1.0):
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
        root=jax.vmap(root_fn, (None, 0))(state, jnp.ones(1)),
        recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
        num_simulations=8,
        max_depth=4,
        max_num_considered_actions=params["num_actions"],
    )

    best_action = select_action(policy_output, selection_rule, epsilon, temperature, c)
    return policy_output, best_action

def gather_data(env, agent, buffer_state, state, timestep, num_steps, subkey, selection_rule="greedy", epsilon=0.1, temperature=1.0, c=1.0):
    batches = []
    for batch in range(params["num_batches"]):
        states, action_list, rewards = [], [], []

        for step in range(num_steps):
            print(f'\rGathered Step {step} / {num_steps}', end='')

            actions, best_action = get_actions(agent, state, subkey, selection_rule, epsilon, temperature, c)
            action_weights = actions.action_weights[0]

            buffer_state = buffer.add(buffer_state, timestep)

            next_state, next_timestep = env.step(state, best_action)

            states.append(state.board)
            action_list.append(action_weights)
            rewards.append(timestep.reward)

            state = next_state
            timestep = next_timestep

            if timestep.step_type == StepType.LAST:
                state, timestep = jax.jit(env.reset)(subkey)

        batches.append({
            "states": jnp.array(states),
            "actions": jnp.array(action_list),
            "rewards": jnp.array(rewards)
        })

    return batches

def train(agent, buffer, batches):
    for batch in batches:
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        returns = rewards

        value_loss = agent.update_value(states, returns)
        policy_loss = agent.update_policy(states, actions)

        print(f"Policy Loss: {policy_loss}, Value Loss: {value_loss}, Returns: {jnp.sum(returns)}")

def run_trials(env, agent, selection_rule, epsilon, temperature, c, num_trials):
    rewards_per_trial = []

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials} for {selection_rule}")
        key = jax.random.PRNGKey(params["seed"] + trial)
        rng_key, subkey = jax.random.split(key)
        state, timestep = jax.jit(env.reset)(key)
        buffer_state = buffer.init(timestep)

        rewards = []
        for i in range(params['num_steps']):
            actions, best_action = get_actions(agent, state, subkey, selection_rule, epsilon, temperature, c)
            state, timestep = env_step(state, best_action)
            rewards.append(timestep.reward)

        total_reward = sum(rewards)
        rewards_per_trial.append(total_reward)

    return rewards_per_trial

if __name__ == "__main__":
    env = jumanji.make(params["env_name"])
    env = AutoResetWrapper(env)
    params["num_actions"] = env.action_spec.num_values
    print(f"Action Spec: {env.action_spec}")
    agent = AlphaZero(params, env)

    key = jax.random.PRNGKey(params["seed"])
    rng_key, subkey = jax.random.split(key)

    state, timestep = jax.jit(env.reset)(key)

    buffer = fbx.make_flat_buffer(
        max_length=params['buffer_max_length'],
        min_length=params['buffer_min_length'],
        sample_batch_size=params['num_batches']
    )
    buffer_state = buffer.init(timestep)

    selection_rules = ["greedy", "epsilon_greedy", "softmax", "ucb"]
    epsilon = 0.1
    temperature = 1.0
    c = 1.0

    results = {}

    for selection_rule in selection_rules:
        print(f"\nTrying selection rule: {selection_rule}")

        batches = gather_data(env, agent, buffer_state, state, timestep, params["num_steps"], subkey, selection_rule, epsilon, temperature, c)

        if buffer.can_sample:
            print("Starting training...")
            for epoch in range(params["num_epochs"]):
                print(f"Training Epoch: {epoch + 1}")
                train(agent, buffer, batches)
        else:
            print("Not enough data to start training.")

        rewards_per_trial = run_trials(env, agent, selection_rule, epsilon, temperature, c, params["num_trials"])
        results[selection_rule] = rewards_per_trial

    for rule, rewards in results.items():
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"Selection Rule: {rule}, Average Reward: {avg_reward}, Std Dev: {std_reward}")

    fig, ax = plt.subplots()
    ax.boxplot([results[rule] for rule in selection_rules], labels=selection_rules)
    ax.set_title("Reward Distribution for Different Action Selection Rules")
    ax.set_xlabel("Action Selection Rule")
    ax.set_ylabel("Total Reward")
    plt.show()
