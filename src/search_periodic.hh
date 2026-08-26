/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARBOL_KD_SEARCH_PERIODIC_HH
#define ARBOL_KD_SEARCH_PERIODIC_HH

#include "kd_node.hh"
#include "search.hh"
#include "trait.hh"
#include "type.hh"

#include <Eigen/Core>

namespace arbol_kd {
template <typename T, size_t Dim>
using Box = Eigen::Matrix<T, Dim, 2>;

/**
 * @brief Calculate the image distance between two points in a periodic box.
 *
 * @tparam T Type of the point coordinates (e.g., float, double)
 * @tparam Dim Dimension of the point (e.g., 2D, 3D)
 * @param p1 A point in the periodic space
 * @param p2 Another point in the periodic space
 * @param box Periodic box defined by two corners (min and max)
 * @return KDNode<T, Dim>::Type The image distance vector between p1 and p2
 */
template <typename T, size_t Dim>
inline typename KDNode<T, Dim>::Type img_dist_impl(const typename KDNode<T, Dim>::Type &p1,
                                                   const typename KDNode<T, Dim>::Type &p2,
                                                   const Box<T, Dim> &box) {
  const auto &delta = p1 - p2;
  const auto &box_len = box.col(1) - box.col(0);
  return delta - (((delta).array() / box_len.array()).round().array() * box_len.array()).matrix();
}

/**
 * @brief Clamp a point to the periodic image within a box.
 *
 * @tparam T Type of the point coordinates (e.g., float, double)
 * @tparam Dim Dimension of the point (e.g., 2D, 3D)
 * @param point point to be clamped
 * @param box periodic box defined by two corners (min and max)
 * @return KDNode<T, Dim>::Type clamped point
 */
template <typename T, size_t Dim>
inline typename KDNode<T, Dim>::Type
clamp2periodic_box_impl(const typename KDNode<T, Dim>::Type &point, const Box<T, Dim> &box) {

  const auto &box_min = box.col(0);
  const auto &box_len = box.col(1) - box.col(0);
  const auto &mapped =
      point.array() -
      (((point - box_min).array() / box_len.array()).array().floor().array() * box_len.array())
          .array();
  return mapped;
}

/**
 * @brief Clamp a point to the nearest periodic "image" relative to a query point.
 *
 * @tparam T Type of the point coordinates (e.g., float, double)
 * @tparam Dim Dimension of the point (e.g., 2D, 3D)
 * @param point point to be clamped
 * @param query query point to which the clamping is relative
 * @param box periodic box defined by two corners (min and max)
 * @return KDNode<T, Dim>::Type clamped point
 */
template <typename T, size_t Dim>
inline typename KDNode<T, Dim>::Type clamp2nearest_impl(const typename KDNode<T, Dim>::Type &point,
                                                        const typename KDNode<T, Dim>::Type &query,
                                                        const Box<T, Dim> &box) {
  const auto &delta = img_dist_impl<T, Dim>(point, query, box);
  return query + delta;
}

/**
 * @brief Get the relevant images of a point in a periodic box. In case of a 2D box, this
 *  will return the point itself and its 8 images (including the original point). In case
 *  of a 3D box, this will return the point itself and its 26 images (including the original
 *  point).
 *
 * @tparam T Type of the point coordinates (e.g., float, double)
 * @tparam Dim Dimension of the point (e.g., 2D, 3D)
 * @param point point for which to get the relevant images
 * @param box periodic box defined by two corners (min and max)
 * @return std::vector<typename KDNode<T, Dim>::Type> A vector of relevant images of the point.
 */
template <typename T, size_t Dim>
inline auto get_relevant_impl(const typename KDNode<T, Dim>::Type &point, const Box<T, Dim> &box) {
  const auto &box_len = box.col(1) - box.col(0);
  const auto &real_point = clamp2periodic_box_impl<T, Dim>(point, box);
  std::vector<typename KDNode<T, Dim>::Type> images;
  images.reserve(static_cast<size_t>(std::pow(3, Dim)));
  images.emplace_back(real_point);
  for (i64 i = 0; i < Dim; ++i) {
    auto disp = Eigen::Vector<T, Dim>::Zero().eval();
    disp[i] = box_len[i];
    for (auto &image : images) {
      images.emplace_back(image + disp);
      images.emplace_back(image - disp);
    }
  }
  return images;
}

/**
 * @brief Perform a k-nearest neighbor search in a periodic space.
 *
 * @tparam T Type of the point coordinates (e.g., float, double)
 * @tparam Dim  Dimension of the point (e.g., 2D, 3D)
 * @param tree The KD-tree to search in.
 * @param query The query point for which to find the k-nearest neighbors.
 * @param box The periodic box defined by two corners (min and max).
 * @param k Number of nearest neighbors to find.
 * @return BufferTrait<T, Dim>::Type A buffer containing the k-nearest neighbors found in the
 * periodic space.
 */
template <typename T, size_t Dim, class ExecutionPolicy = seq>
inline auto k_nearest_periodic_impl(const typename KDNode<T, Dim>::Ptr &tree,
                                    const typename KDNode<T, Dim>::Type &query,
                                    const Box<T, Dim> &box, const size_t k) {
  // Put root into the buffer to start the search
  typename BufferTrait<T, Dim>::Type buf;
  buf.emplace_back(tree, (tree->point - query).squaredNorm());

  for (const auto &phantom_query : get_relevant_impl<T, Dim>(query, box)) {
    // Perform the search for each phantom query
    node_query_impl<T, Dim, ExecutionPolicy>(tree, phantom_query, 0, k, buf);
  }

  return buf;
}
} // namespace arbol_kd

#endif // ARBOL_KD_SEARCH_PERIODIC_HH