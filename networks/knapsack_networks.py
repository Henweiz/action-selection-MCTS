import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import compact

class KnapsackPolicyNetwork(nn.Module):
    num_actions: int # Number of possible actions

    @compact
    def __call__(self, inputs):
        weights = inputs[:, 0, :]
        values = inputs[:, 1, :]
        packed_items = inputs[:, 2, :]
        action_mask = inputs[:, 3, :]
        # Process weights through two dense layers
        w = nn.Dense(features=64)(weights)
        # w = nn.relu(w)
        # w = nn.Dense(features=32)(w)
        w = nn.relu(w)
        
        # Process values through two dense layers
        v = nn.Dense(features=64)(values)
        # v = nn.relu(v)
        # v = nn.Dense(features=32)(v)
        v = nn.relu(v)
        
        # Concatenate transformed weights and values with packed_items
        input_features = jnp.concatenate([w, v, packed_items], axis=-1)
        
        # First fully connected layer after concatenation
        x = nn.Dense(features=128)(input_features)
        x = nn.relu(x)
        
        # Second fully connected layer
        # x = nn.Dense(features=64)(x)
        # x = nn.relu(x)
        
        # Third fully connected layer
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        
        # Output layer to produce logits for each item
        logits = nn.Dense(features=self.num_actions)(x)
        
        # Mask the logits to ensure only available actions are considered
        # masked_logits = jnp.where(action_mask, logits, -jnp.inf)
        
        # Return the probability distribution over actions using softmax
        action_probs = nn.softmax(logits, axis=-1)
        
        return action_probs

class KnapsackValueNetwork(nn.Module):
    @compact
    def __call__(self, inputs):
        # Split the inputs along the second dimension into weights, values, packed_items, and action_mask
        weights = inputs[:, 0, :]
        values = inputs[:, 1, :]
        packed_items = inputs[:, 2, :]
        action_mask = inputs[:, 3, :]
        # Process weights through two dense layers
        w = nn.Dense(features=64)(weights)
        # w = nn.relu(w)
        # w = nn.Dense(features=32)(w)
        w = nn.relu(w)
        
        # Process values through two dense layers
        v = nn.Dense(features=64)(values)
        # v = nn.relu(v)
        # v = nn.Dense(features=32)(v)
        v = nn.relu(v)
        
        # Concatenate transformed weights and values with packed_items
        input_features = jnp.concatenate([w, v, packed_items.astype(jnp.float32)], axis=-1)
        
        # First fully connected layer after concatenation
        x = nn.Dense(features=128)(input_features)
        x = nn.relu(x)
        
        # Second fully connected layer
        # x = nn.Dense(features=64)(x)
        # x = nn.relu(x)
        
        # Third fully connected layer
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        
        # Output layer to produce logits for each item
        x = nn.Dense(features=1)(x)
        
        return x 