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



def train(timesteps, agent: AlphaZero, env, last):
    key = jax.random.PRNGKey(params["seed"])
    state, timestep = jax.jit(env.reset)(key)
    total_policy_loss = 0
    total_value_loss = 0
    total_steps = 0

    def root_fn(state, _):
        root = mctx.RootFnOutput(
            prior_logits=agent.get_actions(state),
            value=agent.get_value(state),
            embedding=state,
        )
        return root

    for global_step in range(timesteps):
        print(f"Step {global_step}")
        rng_key, subkey = jax.random.split(key)


        policy_output = mctx.gumbel_muzero_policy(
            params=agent,
            rng_key=subkey,
            root=jax.vmap(root_fn, (None, 0))(state, jnp.ones(params["sample_batch_size"])),
            recurrent_fn=jax.vmap(recurrent_fn, (None, None, 0, 0)),
            num_simulations=8,
            max_depth=4,
            max_num_considered_actions=params["num_actions"],
        )
        print(policy_output)

        action = policy_output.action[0]
        action_weights = policy_output.action_weights
        #print(policy_output)

        next_state, next_timestep = env_step(state, action)
        policy_loss, value_loss = agent.update(
            timestep, action_weights
        )  # Pass actions as batch
        total_policy_loss += policy_loss
        total_value_loss += value_loss

        state = next_state
        timestep = next_timestep

        # Renders the environment during the last training cycle.
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
