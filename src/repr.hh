/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARBOL_KD_REPR_HH
#define ARBOL_KD_REPR_HH

#include "kd_node.hh"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>

namespace arbol_kd {
// Helper function to count nodes in a tree
template <typename T, std::size_t Dim>
inline std::size_t count_nodes(KDNode<T, Dim> const *const node_ptr) {
  if (!node_ptr) {
    return 0;
  }
  return 1 + count_nodes(node_ptr->left.get()) + count_nodes(node_ptr->right.get());
}

// Helper function to get tree depth
template <typename T, std::size_t Dim>
inline std::size_t tree_depth(KDNode<T, Dim> const *const node_ptr) {
  if (!node_ptr) {
    return 0;
  }
  return 1 + std::max(tree_depth(node_ptr->left.get()), tree_depth(node_ptr->right.get()));
}

template <typename T, std::size_t Dim>
inline std::string repr_impl(KDNode<T, Dim> const *const node_ptr, const std::string &prefix = "",
                             std::size_t current_depth = 0,
                             std::size_t max_depth = std::numeric_limits<std::size_t>::max()) {
  if (!node_ptr) {
    return "";
  }

  std::ostringstream oss;
  // If we've reached max depth, indicate there's more but don't recurse further
  if (current_depth >= max_depth) {
    if (node_ptr->left || node_ptr->right) {
      oss << " [...]";
    }
    return oss.str();
  }
  oss << std::fixed
      << node_ptr->point.transpose().format(
             Eigen::IOFormat(8, Eigen::DontAlignCols, ", ", "\n", "[", "]"));

  std::string left_prefix = "└─", left_indent = "   ";
  if (node_ptr->left && node_ptr->right) {
    left_prefix = "├─";
    if (node_ptr->left->left || node_ptr->left->right) {
      left_indent = "│  ";
    }
  }

  if (node_ptr->left) {
    oss << std::endl;
    oss << prefix << left_prefix
        << repr_impl(node_ptr->left.get(), prefix + left_indent, current_depth + 1, max_depth);
  }

  if (node_ptr->right) {
    oss << std::endl;
    oss << prefix << "└─"
        << repr_impl(node_ptr->right.get(), prefix + "  ", current_depth + 1, max_depth);
  }

  return oss.str();
}

template <typename T, std::size_t Dim>
inline std::string repr(KDNode<T, Dim> const *const node_ptr, const std::string &prefix = "",
                        std::size_t max_depth = 5) {
  if (!node_ptr) {
    return "";
  }

  std::size_t total_nodes = count_nodes(node_ptr);
  std::size_t total_depth = tree_depth(node_ptr);

  std::ostringstream oss;

  // Add summary info if the tree is large
  if (total_nodes > 20) {
    oss << "KDTree: " << total_nodes << " nodes, depth " << total_depth << std::endl;
    if (max_depth < total_depth) {
      oss << "(Showing up to depth " << max_depth << ")" << std::endl;
    }
  }

  oss << repr_impl(node_ptr, prefix, 0, max_depth);
  return oss.str();
}

} // namespace arbol_kd

#endif // ARBOL_KD_REPR_HH
