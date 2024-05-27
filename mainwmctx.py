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
    "env": "Game2048-v1",
    "seed": 42,
    "lr": 0.01,
    "num_epochs": 100,
    "num_steps": 100,
    "num_actions": 4,
    "buffer_max_length": 5000,  # Set a large buffer size
    "buffer_min_length": 3,  # Set minimum transitions before sampling
    "sample_batch_size": 1,  # Batch size for sampling from the buffer
}


@jax.jit
def env_step(timestep, action): 
    next_state, next_timestep = env.step(timestep, action)
    return next_state, next_timestep
# def create_mctx_model():
#     num_states = 3
#     num_actions = params['num_actions']

#     transition_matrix = jnp.array([
#         [1, 2],
#         [0, 0],
#         [0, 0]
#     ], dtype=jnp.int32)
#     rewards = jnp.array([
#         [1, 0],
#         [0, 0],
#         [0, 1]
#     ], dtype=jnp.float32)
#     discounts = jnp.ones_like(rewards)
#     values = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
#     prior_logits = jnp.zeros((num_states, num_actions), dtype=jnp.float32)

#     def root_fn(batch_size: int):
#         root_state = 0
#         root = mctx.RootFnOutput(
#             prior_logits=jnp.full([batch_size, num_actions], prior_logits[root_state]),
#             value=jnp.full([batch_size], values[root_state]),
#             embedding=jnp.full([batch_size], root_state, dtype=jnp.int32),
#         )
#         return root

#     def recurrent_fn(params, rng_key, action, embedding):
#         del params, rng_key
#         # batch_size = action.shape[0]
#         recurrent_fn_output = mctx.RecurrentFnOutput(
#             reward=rewards[embedding, action],
#             discount=discounts[embedding, action],
#             prior_logits=prior_logits[embedding],
#             value=values[embedding]
#         )
#         next_embedding = transition_matrix[embedding, action]
#         return recurrent_fn_output, next_embedding

#     return root_fn, recurrent_fn


def recurrent_fn(params: AlphaZero, rng_key, action, embedding):
    """One simulation step in MCTS."""
    del rng_key
    agent = params
    env = params.env
    # print(f"Embedding shape: {embedding.shape}")
    # print(f"Action shape: {action.shape}")
    new_embedding, timestep = env_step(embedding, action)
    # state = jax.vmap(lambda e: e.canonical_observation())(env)
    # prior_logits, value = jax.vmap(lambda a, s: (a.policy_apply_fn(a.policy_train_state.params, s), a.value_apply_fn(a.value_train_state.params, s)), in_axes=(None, 0))(agent, embedding)
    prior_logits = agent.get_actions(new_embedding)
    value = agent.get_value(new_embedding)
    discount = timestep.discount
    # terminated = timestep.step_type == jax.StepType.LAST
    # assert value.shape == terminated.shape
    # value = jnp.where(terminated, 0.0, value)
    # assert discount.shape == terminated.shape
    # discount = jnp.where(terminated, 0.0, discount)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=timestep.reward,
        discount=discount,
        prior_logits=prior_logits,
        value=value,
    )
    return recurrent_fn_output, new_embedding


# def root_fn(env, rng_key):
#     return mctx.RootFnOutput(
#         prior_logits =
#     )


def train(timesteps, agent: AlphaZero, env, last):
    key = jax.random.PRNGKey(params["seed"])
    state, timestep = jax.jit(env.reset)(key)
    total_policy_loss = 0
    total_value_loss = 0
    total_steps = 0
    # batch_size = 1  # Define batch_size here

    # root_fn, recurrent_fn = create_mctx_model()

    for global_step in range(timesteps):
        print(f"Step {global_step}")
        rng_key, subkey = jax.random.split(key)
        # root = root_fn(batch_size=batch_size).replace(
        #     embedding=jnp.full([batch_size], 0, dtype=jnp.int32))  # Initial embedding
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
            root=jax.vmap(root_fn, (None, 0))(state, jnp.ones(params["sample_batch_size"])),
            recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
            num_simulations=32,
            max_depth=10,
            max_num_considered_actions=params["num_actions"],
        )

        action = policy_output.action[0]

        next_state, next_timestep = env.step(state, action)
        policy_loss, value_loss = agent.update(
            timestep, jnp.expand_dims(action, axis=0)
        )  # Pass actions as batch
        total_policy_loss += policy_loss
        total_value_loss += value_loss

        state = next_state
        timestep = next_timestep

        if last:
            env.render(state)
        total_steps = global_step
        if timestep.step_type == StepType.LAST:
            break

    print(f"Avg policy loss: {total_policy_loss / total_steps}")
    print(f"Avg value loss: {total_value_loss / total_steps}")


if __name__ == "__main__":
    env = jumanji.make(params["env"])
    env = AutoResetWrapper(env)
    params["num_actions"] = env.action_spec.num_values
    agent = AlphaZero(params, env)

    for epoch in range(params["num_epochs"]):
        print(f"Current epoch: {epoch}")
        if epoch == params["num_epochs"] - 1:
            train(params["num_steps"], agent, env, True)
        else:
            train(params["num_steps"], agent, env, False)
