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
    preamble = "random" if params["random"] else ""
    wandb.init(
        project="action-selection-mcts",
        name=f"{preamble}{env_short_names[params['env_name']]}_{params['policy']}_sim{params['num_simulations']}_seed{params['seed']}",
        config=relevant_params)

