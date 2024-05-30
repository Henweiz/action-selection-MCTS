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
    "num_epochs": 10,
    "num_steps": 300,
    "num_actions": 4,
    "buffer_max_length": 5000,  # Set a large buffer size
    "buffer_min_length": 20,  # Set minimum transitions before sampling
    "sample_batch_size": 10,  # Batch size for sampling from the buffer
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

def gather_data(env, agent, buffer_state, state, timestep, num_steps):

    def select_action():

        def root_fn(state, _):
            root = mctx.RootFnOutput(
                prior_logits=agent.get_actions(state),
                value=agent.get_value(state),
                embedding=state,
            )
            return root

        # TODO change
        policy_output = mctx.gumbel_muzero_policy(
            params=agent,
            rng_key=subkey,
            root=jax.vmap(root_fn, (None, 0))(state, jnp.ones(params["sample_batch_size"])),
            recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
            num_simulations=8,
            max_depth=4,
            max_num_considered_actions=params["num_actions"],
        )
        return policy_output.action[0]


    for step in range(num_steps):
        print(f'\rGathered Step {step} / {num_steps}', end='')  # Use \r to return to the start of the line

        action = select_action()
        buffer_state = buffer.add(buffer_state, timestep)

        next_state, next_timestep = env.step(state, action)

        # buffer.add(state, action, timestep.reward, next_state, timestep.step_type == StepType.LAST)
        state = next_state
        timestep = next_timestep

        if timestep.step_type == StepType.LAST:
            state, timestep = env.reset()

def train_from_buffer(agent, buffer, num_batches, batch_size):
    for _ in range(num_batches):
        batch = buffer.sample(batch_size).experience

        states = batch.first.observation.board
        rewards = batch.second.reward

        returns = rewards

        value_loss = agent.update_value(states, returns)

        policy_loss = agent.update_policy(states, actions)

        return policy_loss, value_loss



if __name__ == "__main__":
    env = jumanji.make(params["env_name"])
    env = AutoResetWrapper(env)
    params["num_actions"] = env.action_spec.num_values
    agent = AlphaZero(params, env)

    key = jax.random.PRNGKey(params["seed"])
    rng_key, subkey = jax.random.split(key)

    state, timestep = jax.jit(env.reset)(key)

    buffer = fbx.make_flat_buffer(
        max_length=params['buffer_max_length'],
        min_length=params['buffer_min_length'],
        sample_batch_size=params['sample_batch_size']
    )
    buffer_state = buffer.init(timestep)

    # Data Gathering Phase
    gather_data(env, agent, buffer_state, state, timestep, params["num_steps"])

    # Check if buffer has enough samples to start learning
    if buffer.can_sample:
        print("Starting training...")
        # Learning Phase
        for epoch in range(params["num_epochs"]):
            print(f"Training Epoch: {epoch + 1}")
            train_from_buffer(agent, buffer, params["num_batches"], params["batch_size"])
    else:
        print("Not enough data to start training.")

