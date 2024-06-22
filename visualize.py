import chex
import mctx
import numpy as np

# TODO clean up and comment

def get_max_tree_depth(tree: mctx.Tree):
    # chex.assert_rank(tree.node_values, 2)

    depths = np.array([])
    # Add root
    # Add all other nodes and connect them up.
    for batch in range(tree.children_index.shape[0]):
        for step in range(tree.children_index.shape[1]):
            max_depth = 0
            depth_dict = dict()
            depth_dict[0] = 0
            for node_i in range(tree.num_simulations):
                for a_i in range(tree.num_actions):
                    # Index of children, or -1 if not expanded
                    children_i = tree.children_index[batch, step, 0, node_i, a_i]
                    if children_i >= 0:
                        depth_dict[children_i.item()] = depth_dict[node_i] + 1
                        max_depth = max(max_depth, depth_dict[children_i.item()])
            depths = np.append(depths, [max_depth])
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    q25 = np.quantile(depths, 0.25)
    q50 = np.quantile(depths, 0.5)
    q75 = np.quantile(depths, 0.75)
    mean = np.mean(depths)
    std = np.std(depths)
    print("TREE DEPTH STATS: ")
    print(f"Mean: {mean}, std: {std}")
    print(f"min: {min_depth}, 25%: {q25}, 50%: {q50}, 75%: {q75}, max: {max_depth}")

    # return max_depth
