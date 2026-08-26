/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARBOL_KD_BUILD_HH
#define ARBOL_KD_BUILD_HH

#include "kd_node.hh"

#include <cassert>

namespace arbol_kd {
template <typename T, std::size_t Dim, typename Iterator>
typename KDNode<T, Dim>::Ptr build_tree_impl(const Iterator &begin, const Iterator &end,
                                             const size_t depth) {
  if (begin == end) {
    return nullptr;
  }
  assert(std::distance(begin, end) > 0);

  // Sort the points based on the current depth
  const auto &sort_fn = [depth](const auto &a, const auto &b) { return (a[depth] < b[depth]); };
  const auto &mid_dist = std::distance(begin, end) * 0.5;
  std::nth_element(begin, begin + mid_dist, end, sort_fn);
  const auto next_depth{(depth + 1) % Dim};

  const auto mid{std::next(begin, mid_dist)};

  auto node = std::make_shared<KDNode<T, Dim>>();
  node->point = *mid;

  node->left = build_tree_impl<T, Dim>(begin, mid, next_depth);
  node->right = build_tree_impl<T, Dim>(std::next(mid), end, next_depth);
  return node;
}
} // namespace arbol_kd

#endif // ARBOL_KD_BUILD_HH
