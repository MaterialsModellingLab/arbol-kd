# Copyright 2025 WATANABE Aoi
# SPDX-License-Identifier: Apache-2.0
"""Linear search implementation for benchmarking purposes."""

import numba
import numpy as np


def setup_coordinates(n, dim):
    """Setup random coordinates for benchmarking."""
    points = np.random.rand(n, dim)
    query = np.random.rand(dim)
    box = np.array([[0.0, 1.0]] * dim)
    return points, query, box


@numba.njit(cache=True)
def lin_k_nearest(points, query, box, k):
    """Perform a linear search for the k nearest neighbors."""
    box_len = box[:, 1] - box[:, 0]
    N = points.shape[0]
    dist_sq = np.empty(N)

    for i in range(N):
        delta = points[i] - query
        for j in range(3):
            delta[j] -= np.round(delta[j] / box_len[j]) * box_len[j]
        dist_sq[i] = delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2

    idx = np.argsort(dist_sq)[:k]
    return points[idx]


def test_lin_k_nearest():
    """Assert lin_search and kd_search provides the same results."""
    import arbol_kd as akd

    points, query, box = setup_coordinates(100, 3)
    k = 5

    lin_result = lin_k_nearest(points, query, box, k)
    tree = akd.build_tree_3d(points)
    kd_result = akd.k_nearest_periodic(tree, query, box, k)

    np.testing.assert_array_equal(lin_result, kd_result)
