import wandb
import jax.numpy as jnp

env_short_names = {
 "Game2048-v1": "2048",
    "Knapsack-v1": "Knapsack",
    "Maze-v0": "Maze",
    "Snake-v1": "Snake",
}

def init_wandb(params):
    if params["run_in_kaggle"]:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        wandb_api = user_secrets.get_secret("wandb_api")
    else:
        import os
        wandb_api = os.environ['WANDB_API_KEY']

    relevant_params = {k: v for k, v in params.items() if k not in
                      ["maze_size", "agent", "num_actions", "obs_spec", "run_in_kaggle", "logging", "buffer_max_length", "buffer_min_length"]}

    wandb.login(key=wandb_api)
    wandb.init(
        project="action-selection-mcts",
        name=f"{env_short_names[params['env_name']]}_{params['policy']}_sim{params['num_simulations']}_seed{params['seed']}",
        config=relevant_params)


def log_rewards(reward, loss, episode, params):
    # Average reward over all batches and steps
    avg = jnp.sum(reward).item() / params["num_batches"]
    if loss is None:
        loss = 0

    # Get average max reward over all batches
    if params["env_name"] == "Game2048-v1":
        avg_max = jnp.sum(jnp.max(reward, axis=1)).item() / params["num_batches"]
        abs_max = jnp.max(reward).item()
        print(f"Episode {episode}: Avg Reward: {avg}, Avg Max Reward: {avg_max}, Abs Max Reward: {abs_max}, Loss: {loss}")

        wandb.log(
            {
                "Total return": avg,
                "Max return": avg_max,
                "Abs Max return": abs_max,
                "Loss": loss,
            }
        )

    else:
        print(f"Episode {episode}: Avg Reward: {avg}, Loss: {loss}")

        wandb.log(
            {
                "Total return": avg,
                "Loss": loss,
            }
        )