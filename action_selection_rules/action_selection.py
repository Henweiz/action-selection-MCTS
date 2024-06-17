import chex
import jax
import mctx

from .solve_trust_region import VariationalKullbackLeibler


def custom_action_selection(
        rng_key: chex.PRNGKey,
        tree: mctx.Tree,
        node_index: chex.Numeric,
        depth: chex.Numeric,
        *,
        pb_c_init: float = 1.25,
        pb_c_base: float = 19652.0,
        qtransform=mctx.qtransform_by_parent_and_siblings,
        selector=VariationalKullbackLeibler()
) -> chex.Array:
    """Returns the action selected for a node index.

    See Appendix B in https://arxiv.org/pdf/1911.08265.pdf for more details.

    Args:
      rng_key: random number generator state.
      tree: _unbatched_ MCTS tree state.
      node_index: scalar index of the node from which to select an action.
      depth: the scalar depth of the current node. The root has depth zero.
      pb_c_init: constant c_1 in the PUCT formula.
      pb_c_base: constant c_2 in the PUCT formula.
      qtransform: a monotonic transformation to convert the Q-values to [0, 1].

    Returns:
      action: the action selected from the given node.
    """
    prior_logits = tree.children_prior_logits[node_index]
    prior_probs = jax.nn.softmax(prior_logits)
    value_score = qtransform(tree, node_index)
    dist = selector(prior_logits, value_score, inv_beta=1.0)

    # Add tiny bit of randomness for tie break

    # Masking the invalid actions at the root.
    #   return masked_argmax(to_argmax, tree.root_invalid_actions * (depth == 0))
    return jax.random.categorical(rng_key, dist)
