from typing import List

import numpy as np
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data


def get_circle_index(data: Data, clockwise: bool = False) -> List[List[int]]:
    """
    Gets a list of indices for every node in data.

    For every node `i`, we create a `circle_i` - list of nodes sorted such that their order corresponds to a real
    3D circle around node `i`. If we imagine node `i` (which represents a directed edge) pointing towards us
    ([0, 0, -1]), then the incoming nodes are sorted counterclockwise (or clockwise if `clockwise` flag is set to True)
    in `circle_i`. As we remove the parallel node from the circle, the length of circle `c_i = max(0, in_degree(i) - 1)`
    for all the nodes.

    Args:
        data: torch geometric graph with `N` nodes (which represent the directed edges) along with the corresponding
            edges (vectors) positions `data.pos` of shape (N, 6).
        clockwise: flag indicating whether to sort nodes clockwise.

    Returns:
        List of N circles.
    """
    circle_index_list = []
    for i in range(len(data.x)):
        in_nodes = data.edge_index[0, data.edge_index[1, :] == i]
        in_nodes = in_nodes[in_nodes != data.parallel_node_index[i]]
        if len(in_nodes) == 0:
            circle_index_list.append([])
            continue
        sorted_index = sort_vectors_in_circle(data.pos[i].numpy(), data.pos[in_nodes].numpy())
        if clockwise:
            sorted_index = np.flip(sorted_index).copy()
        circle_i = in_nodes[sorted_index]
        circle_index_list.append(circle_i.tolist())

    return circle_index_list


def sort_vectors_in_circle(base: np.array, vectors: np.array) -> np.array:
    """
    Sorts vectors in a circle around a base vector. First, we translate and rotate the base and the vectors, so that
    the base starts in [0, 0, 0] and points towards [0, 0, -1]. Then the vectors are projected to xy space and sorted
    counterclockwise.

    Args:
        base: a base of shape (6,) that the `vectors` will be sorted around.
        vectors: vectors of shape (N, 6) to be sorted. All vectors should end on the beginning of the base vector and
            not be parallel to the base.

    Returns:
        Indices of shape (N,) of the sorted vectors.
    """
    vectors = transform_vectors_to_base(base, vectors)
    if not all(np.allclose(v[3:], [0, 0, 0]) for v in vectors):
        raise ValueError('One of the vectors does not ends where the base starts!')
    if any(np.allclose(v[:2], [0, 0]) for v in vectors):
        raise ValueError('One of the vectors is parallel to the base! Sorting is not defined in that case!')
    sorting_values = []
    pxy = vectors[0, :2]
    for i, (xy, z) in enumerate(zip(vectors[:, :2], vectors[:, 2])):
        angle = angle_2d(pxy, xy)
        sorting_values.append((i, angle, -z))

    sorting_values = sorted(sorting_values, key=lambda x: x[1:])
    index = np.array([x[0] for x in sorting_values])
    return index


def transform_vectors_to_base(base: np.array, vectors: np.array) -> np.array:
    """
    Translates and rotates the base and the vectors, so that the base starts in [0, 0, 0] and points towards [0, 0, -1].

    Args:
        base: a base vector of shape (6,) used as a reference vector for translation and rotation.
        vectors: a vectors of shape (N, 6) to be rotated.

    Returns:
        Transformed and rotated `vectors` of shape (N, 6).
    """
    base = base.reshape(2, 3)
    vectors = vectors.reshape(-1, 3)

    # Translate
    translation = -base[0]
    base = base + translation
    vectors = vectors + translation

    base = base[1]

    # Find the angle for rotation in x-axis:
    angle_x = angle_2d(base[1:], np.array([0, -1]))
    rotation_x = np.array([angle_x, 0, 0])

    # Find the angle for rotation in y-axis:
    base = Rotation.from_rotvec(rotation_x).apply(base)
    angle_y = angle_2d(base[np.array([0, 2])], np.array([0, -1]))
    rotation_y = np.array([0, -angle_y, 0])

    # Rotate vectors by found rotation angles:
    vectors = Rotation.from_rotvec(rotation_x).apply(vectors)
    vectors = Rotation.from_rotvec(rotation_y).apply(vectors)

    vectors[np.abs(vectors) < 1e-10] = 0.0
    vectors = vectors.reshape(-1, 6)
    return vectors


def angle_2d(a: np.array, b: np.array) -> np.array:
    """
    Computes angle from 2D vector a to b in a standard (counterclockwise) direction.

    Args:
        a: a 2D vector of shape (2,).
        b: a 2D vector of shape (2,).
    Returns:
        Angle in radians.
    """
    if a.shape != b.shape or a.shape != (2,):
        raise ValueError('You must provide 2D vectors!')
    dot = a[0] * b[0] + a[1] * b[1]
    det = a[0] * b[1] - a[1] * b[0]
    angle = np.arctan2(det, dot)
    return np.mod(angle, 2 * np.pi)
