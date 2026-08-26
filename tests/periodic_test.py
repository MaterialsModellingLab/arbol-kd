# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

import arbol_kd as akd
import numpy as np
import pytest


def test_k_nearest_periodic_2d():
    """
    Test the k-nearest neighbors search in a 2D KDTree with periodic boundary conditions.
    """
    # Create a KDTree with some points
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    box = np.array([[0.0, 2.1], [0.0, 2.1]])
    tree = akd.build_tree_2d(points)

    # Query for the k-nearest neighbors of a point
    query = np.array([1.2, 1.2])
    k = 3
    actual = akd.k_nearest_periodic(tree, query, box, k)

    # Check if the nearest neighbors are correct
    expect = np.array([[1.0, 1.0], [3.0, 3.0], [2.0, 2.0]])
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"


def test_clamp2periodic_box_2d():
    """
    Test the clamping of periodic boundary conditions in a 2D KDTree.
    """
    # Create a KDTree with some points
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    box = np.array([[0.0, 2.1], [0.0, 2.1]])

    # Query for the nearest neighbor of a point outside the box
    actual = akd.clamp2periodic_box(points, box)

    # Check if the clamped point is correct
    expect = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.9, 0.9]])
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"


def test_clamp2nearest_2d():
    """
    Test the clamping of nearest neighbor search in a 2D KDTree with periodic boundary conditions.
    """
    # Create a KDTree with some points
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    box = np.array([[0.0, 2.1], [0.0, 2.1]])

    # Query for the nearest neighbor of a point outside the box
    query = np.array([2.0, 2.0])
    actual = akd.clamp2nearest(points, query, box)

    # Check if the nearest neighbor is clamped correctly
    expect = np.array([[2.1, 2.1], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"


def test_k_nearest_periodic_3d():
    """
    Test the k-nearest neighbors search in a 3D KDTree with periodic boundary conditions.
    """
    # Create a KDTree with some points
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    box = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])
    tree = akd.build_tree_3d(points)

    # Query for the k-nearest neighbors of a point
    query = np.array([1.2, 1.2, 1.2])
    k = 3
    actual = akd.k_nearest_periodic(tree, query, box, k)

    # Check if the nearest neighbors are correct
    expect = np.array([[1, 1, 1], [3, 3, 3], [2, 2, 2]])
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"


def test_clamp2periodic_box_3d():
    """
    Test the clamping of periodic boundary conditions in a 3D KDTree.
    """
    # Create a KDTree with some points
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    box = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])

    # Query for the nearest neighbor of a point outside the box
    actual = akd.clamp2periodic_box(points, box)

    # Check if the clamped point is correct
    expect = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.9, 0.9, 0.9]]
    )
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"


def test_clamp2nearest_3d():
    """
    Test the clamping of nearest neighbor search in a 3D KDTree with periodic boundary conditions.
    """
    # Create a KDTree with some points
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    box = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])

    # Query for the nearest neighbor of a point outside the box
    query = np.array([2.0, 2.0, 2.0])
    actual = akd.clamp2nearest(points, query, box)

    # Check if the nearest neighbor is clamped correctly
    expect = np.array(
        [[2.1, 2.1, 2.1], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"


if __name__ == "__main__":
    pytest.main([__file__])
