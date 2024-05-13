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
    num_actions: int  # Number of possible actions

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return nn.softmax(x)

class TrainState(train_state.TrainState):
    pass
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
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define the XOR data
data = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = jnp.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoding

# Training function


# Training loop
for _ in range(1000):
    state = update(state, data, labels)

# Testing
predictions = state.apply_fn({'params': state.params}, data)
predicted_classes = jnp.argmax(predictions, axis=1)
true_classes = jnp.argmax(labels, axis=1)

print("Predictions:", predictions)
print("Predicted classes:", predicted_classes)
print("True classes:", true_classes)
