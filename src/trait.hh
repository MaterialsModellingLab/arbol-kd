/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARBOL_KD_TRAIT_HH
#define ARBOL_KD_TRAIT_HH

#include "kd_node.hh"

#include <cstddef>
#include <list>
#include <utility>

namespace arbol_kd {
template <typename T, std::size_t Dim>
struct BufferTrait {
  // Pair of node pointer and distance
  using ElementType = std::pair<typename KDNode<T, Dim>::Ptr, T>;

  // Type of the container to hold the elements
  using ContainerType = std::list<ElementType>;

  // Type of the KDNode
  using Type = ContainerType;
};
} // namespace arbol_kd

#endif // ARBOL_KD_TRAIT_HH