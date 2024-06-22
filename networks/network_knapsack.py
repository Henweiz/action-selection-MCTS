import jax
import jax.numpy as jnp
import chex
import haiku as hk
from jumanji.training.networks.knapsack.actor_critic import (
    make_knapsack_masks,
    make_knapsack_query,
    KnapsackTorso,
)
from jumanji.environments.packing.knapsack.types import Observation
from typing import Tuple

config = {
    "transformer_num_blocks": 6,
    "transformer_num_heads": 8,
    "transformer_key_size": 16,
    "transformer_mlp_units": [512],
}

# TODO add some comments

def value_fn(observation: Observation) -> chex.Array:
    torso = KnapsackTorso(
        transformer_num_blocks=config["transformer_num_blocks"],
        transformer_num_heads=config["transformer_num_heads"],
        transformer_key_size=config["transformer_key_size"],
        transformer_mlp_units=config["transformer_mlp_units"],
        name="torso",
    )
    self_attention_mask, cross_attention_mask = make_knapsack_masks(observation)
    items_features = jnp.concatenate(
        [observation.weights[..., None], observation.values[..., None]], axis=-1
    )
    embeddings = torso(items_features, self_attention_mask)
    query = make_knapsack_query(observation, embeddings)
    cross_attention_block = hk.MultiHeadAttention(
        num_heads=config["transformer_num_heads"],
        key_size=config["transformer_key_size"],
        w_init=hk.initializers.VarianceScaling(1.0),
        name="cross_attention_block",
    )
    cross_attention = cross_attention_block(
        query=query,
        value=embeddings,
        key=embeddings,
        mask=cross_attention_mask,
    ).squeeze(axis=-2)
    values = jnp.einsum("...Tk,...k->...T", embeddings, cross_attention)
    values = values / jnp.sqrt(cross_attention_block.model_size)
    value = values.sum(axis=-1, where=cross_attention_mask.squeeze(axis=(-2, -3)))
    return value


def policy_fn(observation: Observation) -> chex.Array:
    torso = KnapsackTorso(
        transformer_num_blocks=config["transformer_num_blocks"],
        transformer_num_heads=config["transformer_num_heads"],
        transformer_key_size=config["transformer_key_size"],
        transformer_mlp_units=config["transformer_mlp_units"],
        name="torso",
    )
    self_attention_mask, cross_attention_mask = make_knapsack_masks(observation)
    items_features = jnp.concatenate(
        [observation.weights[..., None], observation.values[..., None]], axis=-1
    )
    embeddings = torso(items_features, self_attention_mask)
    query = make_knapsack_query(observation, embeddings)
    cross_attention_block = hk.MultiHeadAttention(
        num_heads=config["transformer_num_heads"],
        key_size=config["transformer_key_size"],
        w_init=hk.initializers.VarianceScaling(1.0),
        name="cross_attention_block",
    )
    cross_attention = cross_attention_block(
        query=query,
        value=embeddings,
        key=embeddings,
        mask=cross_attention_mask,
    ).squeeze(axis=-2)
    logits = jnp.einsum("...Tk,...k->...T", embeddings, cross_attention)
    logits = logits / jnp.sqrt(cross_attention_block.model_size)
    logits = 10 * jnp.tanh(logits)  # clip to [-10,10]
    logits = jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)
    actions = jax.nn.softmax(logits)
    return actions


def forward_fn(inputs) -> Tuple[chex.Array, chex.Array]:
    weights = inputs[:, 0, :].astype(jnp.float32)
    values = inputs[:, 1, :].astype(jnp.float32)
    packed_items = inputs[:, 2, :].astype(jnp.bool)
    action_mask = inputs[:, 3, :].astype(jnp.bool)
    observation = Observation(weights, values, packed_items, action_mask)
    return policy_fn(observation), value_fn(observation)
