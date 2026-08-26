# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.typing import NDArray

class KDNode2D:
    def __init__(self) -> None: ...
    def __repr__(self, max_depth: int = 3) -> str: ...

class KDNode3D:
    def __init__(self) -> None: ...
    def __repr__(self, max_depth: int = 3) -> str: ...

def build_tree_2d(points: NDArray[np.float64]) -> KDNode2D: ...
def build_tree_3d(points: NDArray[np.float64]) -> KDNode3D: ...
def k_nearest(
    tree: KDNode2D | KDNode3D, query: NDArray[np.float64], k: int
) -> NDArray[np.float64]: ...
def v_k_nearest(
    tree: KDNode2D | KDNode3D, queries: NDArray[np.float64], k: int
) -> list[NDArray[np.float64]]: ...
def k_nearest_periodic(
    tree: KDNode2D | KDNode3D,
    query: NDArray[np.float64],
    box: NDArray[np.float64],
    k: int,
) -> NDArray[np.float64]: ...
def v_k_nearest_periodic(
    tree: KDNode2D | KDNode3D,
    queries: NDArray[np.float64],
    box: NDArray[np.float64],
    k: int,
) -> list[NDArray[np.float64]]: ...
def clamp2periodic_box(
    points: NDArray[np.float64], box: NDArray[np.float64]
) -> NDArray[np.float64]: ...
def clamp2nearest(
    points: NDArray[np.float64], query: NDArray[np.float64], box: NDArray[np.float64]
) -> NDArray[np.float64]: ...
def v_clamp2nearest(
    points_list: list[NDArray[np.float64]],
    queries: NDArray[np.float64],
    box: NDArray[np.float64],
) -> list[NDArray[np.float64]]: ...
