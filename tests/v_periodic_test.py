# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

import arbol_kd as akd
import numpy as np


def test_v_k_nearest_periodic_2d():
    """
    Test the k-nearest neighbors search in a 2D KDTree with periodic boundary conditions.
    """
    # Create a KDTree with some points
    points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    box = np.array([[0.0, 2.1], [0.0, 2.1]])
    tree = akd.build_tree_2d(points)

    # Query for the k-nearest neighbors of a point
    queries = np.array([[1.2, 1.2], [1.6, 1.6]])
    k = 3
    actual = akd.v_k_nearest_periodic(tree, queries, box, k)

    # Check if the nearest neighbors are correct
    expect = [
        np.array([[1.0, 1.0], [3.0, 3.0], [2.0, 2.0]]),
        np.array([[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]),
    ]
    assert np.allclose(expect, actual), f"Expected {expect}, got {actual}"


def test_v_clamp2nearest_2d():
    """
    Test the clamping of nearest neighbor search in a 2D KDTree with periodic boundary conditions.
    """
    # Create a KDTree with some points
    points1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    points2 = np.array([[0.9, 0.9], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1]])
    points_list = [points1, points2]
    box = np.array([[0.0, 2.1], [0.0, 2.1]])

    # Query for the nearest neighbor of a point outside the box
    query = np.array([2.0, 2.0])
    actual = akd.v_clamp2nearest(points_list, query, box)

    # Check if the nearest neighbor is clamped correctly
    expect = [
        [[2.1, 2.1], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        [[3.0, 3.0], [1.1, 1.1], [2.1, 2.1], [1.0, 1.0]],
    ]
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"


def test_v_k_nearest_periodic_3d():
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
    queries = np.array([[1.2, 1.2, 1.2], [1.6, 1.6, 1.6]])
    k = 3
    actual = akd.v_k_nearest_periodic(tree, queries, box, k)

    # Check if the nearest neighbors are correct
    expect = [
        np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [2.0, 2.0, 2.0]]),
        np.array([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    ]
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"


def test_v_clamp2nearest_3d():
    """
    Test the clamping of nearest neighbor search in a 3D KDTree with periodic boundary conditions.
    """
    # Create a KDTree with some points
    points1 = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    points2 = np.array(
        [[0.9, 0.9, 0.9], [1.1, 1.1, 1.1], [2.1, 2.1, 2.1], [3.1, 3.1, 3.1]]
    )
    points_list = [points1, points2]
    box = np.array([[0.0, 2.1], [0.0, 2.1], [0.0, 2.1]])

    # Query for the nearest neighbor of a point outside the box
    query = np.array([2.0, 2.0, 2.0])
    actual = akd.v_clamp2nearest(points_list, query, box)

    # Check if the nearest neighbor is clamped correctly
    expect = [
        [[2.1, 2.1, 2.1], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
        [[3.0, 3.0, 3.0], [1.1, 1.1, 1.1], [2.1, 2.1, 2.1], [1.0, 1.0, 1.0]],
    ]
    assert np.allclose(actual, expect), f"Expected {expect}, got {actual}"
