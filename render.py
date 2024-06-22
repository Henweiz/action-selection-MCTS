import jax.numpy as jnp
import jumanji
import numpy as np
from jumanji.environments.logic.game_2048.types import State
import matplotlib


states = jnp.array(np.load('kaggle/working/2048_states_13.npy'))
rewards = jnp.array(np.load('kaggle/working/2048_rewards_13.npy'))
new_states = []
env = jumanji.make("Game2048-v1")

total_reward = 0
for i, (reward, state) in enumerate(zip(rewards, states)):
    total_reward += reward
    state = State(board=state, score=total_reward, step_count=i, action_mask=jnp.array([1, 1, 1, 1]), key=jnp.array([0, 0]))
    new_states.append(state)
    # env.render(state)


env.animate(new_states, interval=100, save_path='kaggle/working/2048_animation.mp4')
