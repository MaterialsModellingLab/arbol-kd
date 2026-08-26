#!/usr/bin/env python3
# Copyright 2025 WATANABE Aoi
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the performance of linear search in 2D space to verify the performance of KD-Tree."""

import pathlib

import arbol_kd as akd
import numpy as np
import pytest
from util import lin_k_nearest, setup_coordinates

K = 100
X_LIST = np.pow(2, np.arange(10, 22)).tolist()


@pytest.mark.parametrize("x", X_LIST)
def test_lin_k_nearest(benchmark, x):
    """Benchmark the performance of linear search in 3D space."""
    points, query, box = setup_coordinates(x, 3)
    kw_args = {"points": points, "query": query, "box": box, "k": K}
    # For JIT compile
    _ = lin_k_nearest(**kw_args)

    # Benchmark
    benchmark.pedantic(
        lin_k_nearest,
        kwargs=kw_args,
        rounds=100,
        iterations=1,
    )


@pytest.mark.parametrize("x", X_LIST)
def test_kd_k_nearest(benchmark, x):
    """Benchmark the performance of KD-Tree search in 3D space."""
    points, query, box = setup_coordinates(x, 3)
    tree = akd.build_tree_3d(points)

    kw_args = {"tree": tree, "query": query, "box": box, "k": K}

    # Benchmark
    benchmark.pedantic(
        akd.k_nearest_periodic,
        kwargs=kw_args,
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
