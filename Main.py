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

params = {
    'env': 'Game2048-v1',
    'seed': 42,
    'lr': 0.05,
    'num_epochs': 100,
    'num_steps': 500,
    'num_actions': 4,
    'buffer_max_length': 5000,  # Set a large buffer size
    'buffer_min_length': 10,  # Set minimum transitions before sampling
    'sample_batch_size': 32  # Batch size for sampling from the buffer
}

@jax.jit
def env_step(timestep, action): 
    next_state, next_timestep = env.step(timestep, action)
    return next_state, next_timestep


def train(timesteps, agent: AlphaZero, env, last):
    key = jax.random.PRNGKey(params['seed'])
    state, timestep = jax.jit(env.reset)(key)
    total_policy_loss = 0 
    total_value_loss = 0 
    total_steps = 0

    for global_step in range(timesteps):
        actions = agent.get_actions(state)
        actions = mask_actions(jnp.squeeze(actions), state.action_mask)
        action = jnp.argmax(actions, axis=-1)
        #print(state)
        #print(f"Actions: {actions}")
        #print(f"Action taken: {action}")
        
        next_state, next_timestep = env_step(state, action)
        #print(f"Reward: {next_timestep.reward}")

        #if global_step % 10 == 0:
        #    print(f"Current Training Step: {global_step}")

        policy_loss, value_loss = agent.update(timestep, actions)
        total_policy_loss += policy_loss
        total_value_loss += value_loss

        # Update the observation
        state = next_state
        timestep = next_timestep
        
        ''' REMOVE THE COMMENT BELOW TO GET THE VISUALS OF THE GAME.'''
        if last:
            env.render(state)
        total_steps = global_step
        if timestep.step_type == StepType.LAST:
            break;
    print(f"Total steps: {total_steps}")
    print(f"Avg policy loss: {total_policy_loss / total_steps}")
    print(f"Avg value loss: {total_value_loss / total_steps}")

# Maskes the action, since we use softmax in our network we can change the masked values to 0. (I think)
def mask_actions(actions, mask):
    return jnp.where(mask, actions, 0)

if __name__ == '__main__':
    env = jumanji.make(params['env'])
    env = AutoResetWrapper(env)
    params['num_actions'] = env.action_spec.num_values
    agent = AlphaZero(params, env)


    for epoch in range(params['num_epochs']):
        print(f"Current epoch: {epoch}")
        if epoch == params['num_epochs']-1:
            train(params['num_steps'], agent, env, True)
        else:
            train(params['num_steps'], agent, env, False)
