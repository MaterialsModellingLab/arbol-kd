# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

import arbol_kd as akd
import numpy as np
import pytest


def test_k_nearest_2d():
    """
    Test the k-nearest neighbors search in a 2D KDTree.
    """
    # Create a KDTree with some points
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    tree = akd.build_tree_2d(points)

    # Case1: Normal query
    query = np.array([1.2, 1.2])
    actual = akd.k_nearest(tree, query, 3)
    expect = np.array([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"

    # Case2: Larger k than number of points
    query = np.array([0.8, 0.8])
    actual = akd.k_nearest(tree, query, 10)
    expect = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"

    # Case3: Query point is just on some points
    query = np.array([1.0, 1.0])
    actual = akd.k_nearest(tree, query, 2)
    expect = np.array([[1.0, 1.0], [2.0, 2.0]])
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"

    # Case4: Query point is just on root points
    query = np.array([2.0, 2.0])
    actual = akd.k_nearest(tree, query, 2)
    expect = np.array([[2.0, 2.0], [1.0, 1.0]])
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"


def test_k_nearest_3d():
    """
    Test the k-nearest neighbors search in a 3D KDTree.
    """
    # Create a KDTree with some points
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    tree = akd.build_tree_3d(points)

    # Case1: Normal query
    query = np.array([1.2, 1.2, 1.2])
    actual = akd.k_nearest(tree, query, 3)
    expect = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"

    # Case2: Larger k than number of points
    query = np.array([0.8, 0.8, 0.8])
    actual = akd.k_nearest(tree, query, 10)
    expect = np.array(
        [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"

    # Case3: Query point is just on some points
    query = np.array([1.0, 1.0, 1.0])
    actual = akd.k_nearest(tree, query, 2)
    expect = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"

    # Case4: Query point is just on root points
    query = np.array([2.0, 2.0, 2.0])
    actual = akd.k_nearest(tree, query, 2)
    expect = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"


if __name__ == "__main__":
    pytest.main([__file__])
