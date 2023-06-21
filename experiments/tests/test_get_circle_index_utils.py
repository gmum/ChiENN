import sys
from typing import List

from math import sqrt
from matplotlib import pyplot as plt

sys.path.append('submodules/ChIRo')
sys.path.append('submodules/tetra_dmpnn')
sys.path.append('.')

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graphgps.dataset.chienn_utils import get_circle_index, transform_vectors_to_base, sort_vectors_in_circle, angle_2d


def compare_3d(vectors, result):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    vectors = vectors.reshape(-1, 3)
    color = ['gray', 'red', 'green', 'gray', 'blue', 'gray', 'purple', 'gray']
    ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], color=color)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    result = result.reshape(-1, 3)
    ax.scatter(result[:, 0], result[:, 1], result[:, 2], color=color)
    plt.show()


@pytest.mark.parametrize('a, b, expected_angle',
                         [([1, 0], [0, 1], np.pi * 0.5),
                          ([0, 1], [1, 0], np.pi * (3 / 2)),
                          ([0, -1], [1, 0], np.pi * 0.5),
                          ([1, 0], [0, -1], np.pi * (3 / 2)),
                          ([1, 0], [-1, 0], np.pi),
                          ([128, 0], [100, 0], 0),
                          ([sqrt(3), 1], [sqrt(2), sqrt(2)], np.pi / 12),
                          ([sqrt(2), sqrt(2)], [sqrt(3), 1], np.pi * (23 / 12))])
def test__angle_2d__works_properly(a: List[float], b: List[float], expected_angle: float):
    a = np.array(a)
    b = np.array(b)
    angle = angle_2d(a, b)

    assert np.isclose(angle, expected_angle)


@pytest.mark.parametrize('base, expected_result',
                         [(np.array([0, 0, 0, -1, 1, 2]), np.array([0., 0., 0., 0., 0., -2.44948974])),
                          (np.array([0, 0, 0, 1, 1, 1]), np.array([0., 0., 0., 0., 0., -1.73205081])),
                          (np.array([0, 0, 0, -1, 1, -1]), np.array([0., 0., 0., 0., 0., -1.73205081]))])
def test__transform_vectors_to_base__works_on_single_example(base: np.array, expected_result: np.array):
    vectors = np.expand_dims(base, 0)
    expected_result = np.expand_dims(expected_result, 0)

    result = transform_vectors_to_base(base, vectors)

    assert np.allclose(result, expected_result)


def test__transform_vectors_to_base__works_on_example_1():
    base = np.array([0, 0, 0, -1, 1, -1])
    vectors = np.array([[0, 0, 0, -1, 1, -1], [-1, -1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, -1, -1, 0, 0, 0]])
    expected_result = np.array([[0., 0., 0., 0., 0., -1.73205081],
                                [-1.63299316, 0., 0.57735027, 0., 0., 0.],
                                [0.81649658, 1.41421356, 0.57735027, 0., 0., 0.],
                                [0.81649658, -1.41421356, 0.57735027, 0., 0., 0.]])

    result = transform_vectors_to_base(base, vectors)
    # compare_3d(vectors, result)

    assert np.allclose(result, expected_result)


def test__transform_vectors_to_base__works_on_example_2():
    base = np.array([0, 0, 0, -1.5, 3.7, 1.1])
    vectors = np.array([[0, 0, 0, -1.5, 3.7, 1.1], [-1, -1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, -1, -1, 0, 0, 0]])
    expected_result = np.array([[0., 0., 0., 0., 0., -4.14125585],
                                [-1.17606859, 1.24350662, 0.26561991, 0., 0., 0.],
                                [1.38250617, 0.67356609, -0.79685973, 0., 0., 0.],
                                [0.48168767, -0.67356609, 1.52127766, 0., 0., 0.]])

    result = transform_vectors_to_base(base, vectors)
    # compare_3d(vectors, result)

    assert np.allclose(result, expected_result)


def test__transform_vectors_to_base__works_on_example_3():
    base = np.array([3, 8, 17, -1.5, 3.7, 1.1])
    vectors = np.array(
        [[3, 8, 17, -1.5, 3.7, 1.1], [-1, -1.43, -1.5, 3, 8, 17], [1, 1.3, 1.3, 3, 8, 17], [0, 1.8, -2, 3, 8, 17]])
    expected_result = np.array([[0., 0., 0., 0., 0., -17.07483528],
                                [1.4967334, -4.27334109, -20.65607042, 0., 0., 0.],
                                [2.52586018, -2.36898531, -16.83413018, 0., 0., 0.],
                                [2.36635388, -1.02481989, -20.04470288, 0., 0., 0.]])

    result = transform_vectors_to_base(base, vectors)
    # compare_3d(vectors, result)
    assert np.allclose(result, expected_result)


def test__sort_vectors_in_circle__works_on_example():
    base = np.array([0, 0, 0, -1, 1, -1])
    vectors = np.array([[-1, -1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, -1, -1, 0, 0, 0]])
    expected_result = np.array([0, 2, 1])

    result = sort_vectors_in_circle(base, vectors)

    assert np.allclose(result, expected_result)


def test__sort_vectors_in_circle__works_on_parallel_vectors():
    base = np.array([0, 0, 0, -1, 1, -1])
    vectors = np.array([[-1, -1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, -1, -1, 0, 0, 0], [0.4, 0.5, 0.4, 0, 0, 0]])
    expected_result = np.array([0, 2, 1, 3])

    result = sort_vectors_in_circle(base, vectors)

    assert np.allclose(result, expected_result)


def test__get_circle_index__works_on_single_out_edge_tetrahedron():
    x = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
    edge_index = torch.tensor([[1, 0], [2, 0], [3, 0]]).T
    pos = torch.tensor([[0, 0, 0, -1, 1, -1], [-1, -1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, -1, -1, 0, 0, 0]])
    data = Data(x=x, edge_index=edge_index, pos=pos)

    expected_result = [[1, 3, 2], [], [], []]
    result = get_circle_index(data, ignore_parallel_node=False)

    assert expected_result == result


def test__get_circle_index__works_on_tetrahedron():
    x = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [0, 0], [1, 1], [2, 2], [3, 3]])
    edge_index = torch.tensor(
        [[1, 4], [2, 4], [3, 4], [0, 5], [2, 5], [3, 5], [0, 6], [1, 6], [3, 6], [0, 7], [1, 7], [2, 7]]).T
    pos = torch.tensor(
        [[-1, 1, -1, 0, 0, 0], [-1, -1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, -1, -1, 0, 0, 0],
         [0, 0, 0, -1, 1, -1], [0, 0, 0, -1, -1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, -1, -1]])
    data = Data(x=x, edge_index=edge_index, pos=pos)

    expected_result = [[], [], [], [], [1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]]
    result = get_circle_index(data, ignore_parallel_node=False)

    assert expected_result == result


def test__get_circle_index__works_on_tetrahedron_clockwise():
    x = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [0, 0], [1, 1], [2, 2], [3, 3]])
    edge_index = torch.tensor(
        [[1, 4], [2, 4], [3, 4], [0, 5], [2, 5], [3, 5], [0, 6], [1, 6], [3, 6], [0, 7], [1, 7], [2, 7]]).T
    pos = torch.tensor(
        [[-1, 1, -1, 0, 0, 0], [-1, -1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, -1, -1, 0, 0, 0],
         [0, 0, 0, -1, 1, -1], [0, 0, 0, -1, -1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, -1, -1]])
    data = Data(x=x, edge_index=edge_index, pos=pos)

    expected_result = [[], [], [], [], [2, 3, 1], [3, 2, 0], [1, 3, 0], [2, 1, 0]]
    result = get_circle_index(data, clockwise=True, ignore_parallel_node=False)

    assert expected_result == result
