/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARBOL_KD_SEARCH_HH
#define ARBOL_KD_SEARCH_HH

#include "kd_node.hh"
#include "paralib.hh"
#include "trait.hh"

#include <cassert>

namespace arbol_kd {
// Forward declaration of the KDNode class template
template <typename T, std::size_t Dim>
void node_query_impl(const typename KDNode<T, Dim>::Ptr &tree,
                     const typename KDNode<T, Dim>::Type &query, const std::size_t depth,
                     const std::size_t k, typename BufferTrait<T, Dim>::Type &buf);
template <typename T, std::size_t Dim, class ExecutionPolicy = seq>
void k_nearest_impl(const typename KDNode<T, Dim>::Ptr &tree,
                    const typename KDNode<T, Dim>::Type &query, const std::size_t depth,
                    const std::size_t k, typename BufferTrait<T, Dim>::Type &buf);

template <typename T, std::size_t Dim, class ExecutionPolicy = seq>
void node_query_impl(const typename KDNode<T, Dim>::Ptr &tree,
                     const typename KDNode<T, Dim>::Type &query, const std::size_t depth,
                     const std::size_t k, typename BufferTrait<T, Dim>::Type &buf) {
  if (!tree) {
    return;
  }
  k_nearest_impl<T, Dim, ExecutionPolicy>(tree, query, depth, k, buf);

  // check if this node already exists in the buffer
  static const ExecutionPolicy policy{};
  auto is_exist_fn = [&tree](const auto &elem) { return elem.first == tree; };
  if (std::find_if(policy, buf.begin(), buf.end(), is_exist_fn) != buf.end()) {
    return;
  }

  const auto dl = (tree->point - query).squaredNorm();
  const auto pair = std::make_pair(tree, dl);
  const auto insert_it =
      std::upper_bound(buf.begin(), buf.end(), pair,
                       [](const auto &a, const auto &b) { return a.second < b.second; });
  if (insert_it != buf.end() || buf.size() < k) {
    buf.insert(insert_it, pair);
  }
  while (buf.size() > k) {
    buf.pop_back();
  }
}

template <typename T, std::size_t Dim, class ExecutionPolicy>
void k_nearest_impl(const typename KDNode<T, Dim>::Ptr &tree,
                    const typename KDNode<T, Dim>::Type &query, const std::size_t depth,
                    const std::size_t k, typename BufferTrait<T, Dim>::Type &buf) {
  if (!tree) {
    return;
  }

  // Buffer cannot be empty since we always insert the root node at the start
  assert(!buf.empty());

  const auto dx = tree->point[depth] - query[depth];
  const auto dx_sq = dx * dx;

  const auto close_branch = (dx >= 0.0) ? tree->left : tree->right;
  const auto away_branch = (dx >= 0.0) ? tree->right : tree->left;

  const auto next_depth = (depth + 1) % Dim;
  node_query_impl<T, Dim, ExecutionPolicy>(close_branch, query, next_depth, k, buf);

  if (dx_sq < buf.back().second || buf.size() < k) {
    node_query_impl<T, Dim, ExecutionPolicy>(away_branch, query, next_depth, k, buf);
  }
}
} // namespace arbol_kd

#endif // ARBOL_KD_SEARCH_HH
