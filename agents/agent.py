import jax
import jax.numpy as jnp
import optax
from networks.network_2048 import PolicyValueNetwork_2048
from flax.training import train_state, checkpoints
import wandb

class Agent: 
    def __init__(self, params):

        self.network = params.get("network", PolicyValueNetwork_2048)(num_actions=params["num_actions"], num_channels=params["num_channels"])
        self.optimizer = optax.adam(params['lr'])
        self.input_shape = self.input_shape_fn(params["obs_spec"])

        self.key = jax.random.PRNGKey(params['seed'])
        
        self.train_state = train_state.TrainState.create(
            apply_fn=self.network.apply,
            params=self.network.init(self.key, jnp.ones((1, *self.input_shape))),
            tx=self.optimizer
        )

        self.net_apply_fn = jax.jit(self.train_state.apply_fn)
        self.grad_fn = jax.value_and_grad(self.loss_fn)

        self.last_mse_losses = []
        self.last_kl_losses = []

    def input_shape_fn(self, observation_spec):
        raise NotImplementedError()

    def get_state_from_observation(self, observation, batched):
        raise NotImplementedError()

    def normalize_rewards(self, r):
        return r

    def reverse_normalize_rewards(self, r):
        return r

    def save(self, path, step):
       checkpoints.save_checkpoint(target=self.train_state, ckpt_dir=path, step=step)


    def loss_fn(self, params, states, actions, returns, episode):
        # KL Loss for policy part of the network:
        probs, values = self.net_apply_fn(params, states)
        # optax expects this to be log probabilities
        log_probs = jnp.log(probs + 1e-9)

        targets = actions

        kl_loss = optax.losses.kl_divergence(log_predictions=log_probs, targets=targets)
        kl_loss = jnp.mean(kl_loss)

        # MSE Loss for value part of the network:
        mse_loss = optax.l2_loss(values.flatten(), returns)
        mse_loss = jnp.mean(mse_loss)

        self.last_mse_losses.append(mse_loss.item())
        self.last_kl_losses.append(kl_loss.item())

        return kl_loss + mse_loss


    def update_fn(self, states, actions, returns, episode):
        returns = self.normalize_rewards(returns)
        loss, grads = self.grad_fn(self.train_state.params, states, actions, returns, episode)
        self.train_state = self.train_state.apply_gradients(grads=grads)
        return loss


    def get_output(self, state):
        mask = state.action_mask

        # the state has to be gotten depending on the environment
        state = self.get_state_from_observation(state, True)

        # forward pass of the network
        actions, value = self.net_apply_fn(self.train_state.params, state)
        actions = jnp.ravel(actions)
        value = jnp.ravel(value)[0]

        # mask and renormalize the actions
        masked_actions = self.mask_actions(actions, mask)
        renormalized_actions = masked_actions / jnp.sum(masked_actions)

        value = self.reverse_normalize_rewards(value)

        return renormalized_actions, value

    def log_losses(self, episode, params):
        wandb.log({
            "kl_loss": sum(self.last_kl_losses) / len(self.last_kl_losses),
            "mse_loss": sum(self.last_mse_losses) / len(self.last_mse_losses),

        }, step=episode*params["num_steps"]*params["num_batches"])
        self.last_kl_losses = []
        self.last_mse_losses = []
        
    
    def mask_actions(self, actions, mask):
        return jnp.where(mask, actions, 0)

