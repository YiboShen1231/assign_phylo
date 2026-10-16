"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """


    D = np.array(distances, dtype=float, copy=True)
    nodes = [Node(str(label), None, 0, None, 0) for label in labels]
    node_count = 0
    while len(nodes) > 2:
        n = len(nodes)
        row_sums = np.sum(D, axis=1)
        Q = (n - 2) * D - row_sums[:, None] - row_sums[None, :]

        np.fill_diagonal(Q, np.inf)

        i, j = np.unravel_index(np.argmin(Q), Q.shape)

        distance_ij = D[i, j]

        left_length = (
            distance_ij / 2
            + (row_sums[i] - row_sums[j]) / (2 * (n - 2))
        )
        right_length = distance_ij - left_length

        node_count += 1
        new_node = Node(
            f"INTERNAL_{node_count}",
            nodes[i],
            float(left_length),
            nodes[j],
            float(right_length),
        )
        remaining = [k for k in range(n) if k not in (i, j)]

        new_distances = [
            (D[i, k] + D[j, k] - distance_ij) / 2
            for k in remaining
        ]

        D = D[np.ix_(remaining, remaining)]

        D = np.pad(D, ((0, 1), (0, 1)), mode="constant")
        D[-1, :-1] = new_distances
        D[:-1, -1] = new_distances
        nodes = [nodes[k] for k in remaining]
        nodes.append(new_node)

    final_distance = float(D[0, 1])

    root = Node(
        "ROOT",
        nodes[0],
        final_distance / 2,
        nodes[1],
        final_distance / 2,
    )

    return root



def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    positions = {}
    leaf_count = 0

    def calculate_positions(node, x):
        nonlocal leaf_count

        if node.left is None and node.right is None:
            y = leaf_count
            leaf_count += 1
            positions[id(node)] = (x, y)
            return y

        child_ys = []

        if node.left is not None:
            left_y = calculate_positions(
                node.left, x + node.left_distance
            )
            child_ys.append(left_y)

        if node.right is not None:
            right_y = calculate_positions(
                node.right, x + node.right_distance
            )
            child_ys.append(right_y)

        y = sum(child_ys) / len(child_ys)
        positions[id(node)] = (x, y)

        return y

    calculate_positions(tree, 0)

    def draw_branches(node):
        x, y = positions[id(node)]

        for child, distance in [
            (node.left, node.left_distance),
            (node.right, node.right_distance),
        ]:
            if child is None:
                continue

            child_x, child_y = positions[id(child)]

            ax.plot([x, child_x], [child_y, child_y],
                    color="black", linewidth=1.5)

            draw_branches(child)

        if node.left is not None and node.right is not None:
            left_y = positions[id(node.left)][1]
            right_y = positions[id(node.right)][1]

            ax.plot([x, x], [left_y, right_y],
                    color="black", linewidth=1.5)

        if node.left is None and node.right is None:
            label_colors = kwargs.get("label_colors", {})
            color = label_colors.get(node.name, "black")
            ax.text(x, y, f"  {node.name}",
                    fontsize=12, va="center", color=color)
        elif kwargs.get("show_confidence", False) and node is not tree:
            if node.confidence is not None:
                ax.text(x, y, f"{node.confidence:.0%}",
                        fontsize=9, color="blue", va="bottom")

    draw_branches(tree)

    ax.set_xlabel("Genetic distance")
    ax.set_yticks([])
    ax.invert_yaxis()

    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    ax.margins(x=0.15, y=0.1)

    return ax



def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    def count_leaves(node):
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1

        left = count_leaves(node.left)
        right = count_leaves(node.right)

        if left > right:
            node.left, node.right = node.right, node.left
            node.left_distance, node.right_distance = node.right_distance, node.left_distance

        return left + right

    count_leaves(tree)


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()