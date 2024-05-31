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

params = {
    "env_name": "Game2048-v1",
    "seed": 42,
    "lr": 0.01,
    "num_epochs": 1,
    "num_steps": 5,
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
        root=jax.vmap(root_fn, (None, 0))(state, jnp.ones(1 )), #params["num_steps"])),
        recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
        num_simulations=8,
        max_depth=4,
        max_num_considered_actions=params["num_actions"],
    )
    return policy_output



def gather_data(env, agent, buffer_state, state, timestep, num_steps):
    batches = []
    for batch in range(params["num_batches"]):

        states, action_list, rewards = [],[],[],

        for step in range(num_steps):
            print(f'\rGathered Step {step} / {num_steps}', end='')

            actions = get_actions(agent, state, subkey)
            best_action = actions.action[0]
            action_weights = actions.action_weights[0]

            buffer_state = buffer.add(buffer_state, timestep)

            next_state, next_timestep = env.step(state, best_action)

            states.append(state.board)
            action_list.append(action_weights)
            rewards.append(timestep.reward)

            # buffer.add(state, action, timestep.reward, next_state, timestep.step_type == StepType.LAST)
            state = next_state
            timestep = next_timestep

            if timestep.step_type == StepType.LAST:
                state, timestep = env.reset()

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

    # Data Gathering Phase
    batches = gather_data(env, agent, buffer_state, state, timestep, params["num_steps"])

    # Check if buffer has enough samples to start learning
    if buffer.can_sample:
        print("Starting training...")
        # Learning Phase
        for epoch in range(params["num_epochs"]):
            print(f"Training Epoch: {epoch + 1}")
            train(agent, buffer, batches)
    else:
        print("Not enough data to start training.")

    rewards = []
    for i in range(params['num_steps']):
        print(f"Step {i}")
        actions = get_actions(agent, state, subkey)
        best_action = actions.action[0]
        state, timestep = env_step(state, best_action)
        rewards.append(timestep.reward)
        env.render(state)

    print(f"Total Reward: {sum(rewards)}")



