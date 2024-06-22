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
        """Get the shape of the states from the observation spec"""
        raise NotImplementedError()

    def get_state_from_observation(self, observation, batched):
        """Get the state from the observation"""
        raise NotImplementedError()

    def normalize_rewards(self, r):
        """Perform any normalization on the rewards before training the network."""
        return r

    def reverse_normalize_rewards(self, r):
        """Reverse the normalization of the rewards for output."""
        return r

    def save(self, path, step, prefix):
        """Save the networks to a given path"""
        checkpoints.save_checkpoint(
            target=self.train_state,
            ckpt_dir=path,
            step=step,
            overwrite=True,
            prefix=prefix
        )
    
    def load(self, path, prefix):
        """Load the networks from a given path"""
        latest = checkpoints.latest_checkpoint(path, prefix=prefix)
        self.train_state = checkpoints.restore_checkpoint(latest, target=self.train_state, prefix=prefix)

    def loss_fn(self, params, states, actions, returns):
        """Compute the loss of the network"""

        # forward pass of the network, get predicted actions and values
        probs, values = self.net_apply_fn(params, states)

        # optax expects this to be log probabilities
        log_probs = jnp.log(probs + 1e-9)

        # KL Loss for policy part of the network:
        kl_loss = optax.losses.kl_divergence(log_predictions=log_probs, targets=actions)
        kl_loss = jnp.mean(kl_loss)

        # MSE Loss for value part of the network:
        mse_loss = optax.l2_loss(values.flatten(), returns)
        mse_loss = jnp.mean(mse_loss)

        # save the losses for logging
        self.last_mse_losses.append(mse_loss.item())
        self.last_kl_losses.append(kl_loss.item())

        return kl_loss + mse_loss


    def update_fn(self, states, actions, returns):
        """Update the network with the given states, actions and returns"""

        # normalize the returns
        returns = self.normalize_rewards(returns)

        # compute the loss and gradients
        loss, grads = self.grad_fn(self.train_state.params, states, actions, returns)

        # apply the gradients and return the loss
        self.train_state = self.train_state.apply_gradients(grads=grads)
        return loss


    def get_output(self, state):
        """Get the output of the network given a state"""
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

        # reverse normalize the values from the network to fit the environment if previously normalized
        value = self.reverse_normalize_rewards(value)

        return renormalized_actions, value

    def log_losses(self, episode, params):
        """Log the losses of the network every episode"""
        if params["logging"]:
            wandb.log({
                "kl_loss": sum(self.last_kl_losses) / len(self.last_kl_losses),
                "mse_loss": sum(self.last_mse_losses) / len(self.last_mse_losses),

            }, step=episode*params["num_steps"]*params["num_batches"])
        self.last_kl_losses = []
        self.last_mse_losses = []
        
    
    def mask_actions(self, actions, mask):
        """Mask the actions with the given mask"""
        return jnp.where(mask, actions, 0)

