/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARBOL_KD_KD_NODE_HH
#define ARBOL_KD_KD_NODE_HH

#include <Eigen/Core>
#include <memory>

namespace arbol_kd {
template <typename T, std::size_t Dim>
struct KDNode {
  using Ptr = std::shared_ptr<KDNode>;
  using Type = Eigen::Vector<T, Dim>;
  Type point;
  KDNode::Ptr left, right;
};
} // namespace arbol_kd

#endif // ARBOL_KD_KD_POINT_HH
