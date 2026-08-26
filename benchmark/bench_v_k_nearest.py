#!/usr/bin/env python3
# Copyright 2025 WATANABE Aoi
# SPDX-License-Identifier: Apache-2.0
"""Benchmark for k-nearest neighbors search using threading."""

import pathlib

import arbol_kd as akd
import numpy as np
import pytest

N = np.pow(2, 10)
K = 100

# Number of queries
X_LIST = np.pow(2, np.arange(2, 12)).tolist()


def setup_coordinates(n, k, dim):
    """Set up random coordinates for benchmarking."""
    points = np.random.rand(n, dim)
    queries = np.random.rand(k, dim)
    box = np.array([[0.0, 1.0]] * dim)
    return points, queries, box


@pytest.mark.parametrize("x", X_LIST)
def test_v_k_nearest_neighbors_serial(benchmark, x):
    points, queries, box = setup_coordinates(N, x, 3)
    tree = akd.build_tree_3d(points)

    def func():
        return np.array(
            [akd.k_nearest_periodic(tree, query, box, k=100) for query in queries]
        )

    benchmark.pedantic(
        func,
        kwargs={},
        rounds=100,
        iterations=1,
    )


@pytest.mark.parametrize("x", X_LIST)
def test_v_k_nearest_neighbors_parallel(benchmark, x):
    points, queries, box = setup_coordinates(N, x, 3)
    tree = akd.build_tree_3d(points)
    benchmark.pedantic(
        akd.v_k_nearest_periodic,
        kwargs={"tree": tree, "queries": queries, "box": box, "k": 100},
        rounds=100,
        iterations=1,
    )


if __name__ == "__main__":
    filename = pathlib.Path(__file__).stem
    pytest.main(
        [
            "-v",
            __file__,
            "--benchmark-only",
            "--benchmark-warmup=True",
            "--benchmark-warmup-iterations=10",
            f"--benchmark-save={filename}",
        ]
    )
