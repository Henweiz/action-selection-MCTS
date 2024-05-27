import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import grad, jit
from flax.training import train_state
import optax

# Define the XOR Policy Network
# class XORNetwork(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(4)(x)
#         x = nn.relu(x)
#         x = nn.Dense(2)(x)
#         x = nn.relu(x)
#         x = nn.Dense(2)(x)
#         return nn.softmax(x)

class PolicyNetwork(nn.Module):
    """A simple policy network that outputs a probability distribution over actions."""

    num_actions: int # Number of possible actions

    @nn.compact
    def __call__(self, x):

        x = jnp.reshape(x, (x.shape[0], -1)) #flatten, do not that we get errors when we do not input batches
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return nn.softmax(x)

'''
@jit
def update(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        return -jnp.mean(jnp.sum(logits * y, axis=1))  # Cross-entropy loss

    grads = grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

# Initialize the network and parameters
key = jax.random.PRNGKey(0)
model = PolicyNetwork(2)
params = model.init(key, jnp.array([[0, 0]]))['params']

# Create training state
tx = optax.adam(learning_rate=0.1)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define the XOR data
data = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = jnp.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoding
'''

# Define the Value Network
class ValueNetwork(nn.Module):
    """A simple value network."""
    #num_outputs: int = 1

    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, (x.shape[0], -1)) # flatten
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
'''
# Initialize the network and parameters
key = jax.random.PRNGKey(1)  # Different key to avoid identical initializations
model = ValueNetwork()
params = model.init(key, jnp.array([[0, 1]]))['params']

# Create training state
tx = optax.adam(learning_rate=0.05)  # Smaller learning rate for demonstration
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define the XOR data (same as before)
data = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Define arbitrary value targets, not one-hot but scalar regression targets
targets = jnp.array([[0.1], [0.8], [0.8], [0.1]])

# Training function
@jit
def update(state, x, y):
    def loss_fn(params):
        values = state.apply_fn({'params': params}, x)
        return jnp.mean((values - y) ** 2)  # Mean squared error loss

    grads = grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

# Training loop
for _ in range(5000):
    state = update(state, data, targets)

# Testing
predicted_values = state.apply_fn({'params': state.params}, data)

print("Predicted values:", predicted_values)
'''