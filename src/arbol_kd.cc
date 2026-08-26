/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

#include "build.hh"
#include "paralib.hh"
#include "repr.hh"
#include "search.hh"
#include "search_periodic.hh"
#include "type.hh"

#include <stdexcept>

#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace arbol_kd {
template <typename T, std::size_t Dim>
inline typename KDNode<T, Dim>::Ptr build_tree(const py::array_t<T> &points) {
  const auto &buf = points.request();
  if (buf.ndim != 2 || buf.shape[1] != Dim) {
    throw std::invalid_argument("Expected shape (N, " + std::to_string(Dim) + ")");
  }
  const std::size_t N = buf.shape[0];
  const T *const ptr = static_cast<const T *>(buf.ptr);

  std::vector<typename KDNode<T, Dim>::Type> in;
  in.reserve(N);
  for (std::size_t i = 0; i < N; ++i) {
    typename KDNode<T, Dim>::Type p;
    std::copy(ptr + i * Dim, ptr + (i + 1) * Dim, p.begin());
    in.emplace_back(p);
  }
  return build_tree_impl<T, Dim>(in.begin(), in.end(), 0);
}

template <typename T, std::size_t Dim, class ExecutionPolicy = seq>
inline auto k_nearest(const typename KDNode<T, Dim>::Ptr tree,
                      const py::EigenDRef<const Eigen::Vector<T, Dim>> query, std::size_t k) {
  // Put root into the buffer to start the search
  typename BufferTrait<T, Dim>::Type buf;
  buf.emplace_back(tree, (tree->point - query).squaredNorm());

  // Perform the search
  k_nearest_impl<T, Dim, ExecutionPolicy>(tree, query, 0, k, buf);
  Eigen::Matrix<T, Eigen::Dynamic, Dim> ret(buf.size(), Dim);
  const auto fn = [](const typename BufferTrait<T, Dim>::ElementType &elem) {
    return elem.first->point;
  };

  static const ExecutionPolicy policy{};
  std::transform(policy, buf.begin(), buf.end(), ret.rowwise().begin(), fn);

  return ret;
}

template <typename T, std::size_t Dim, class ExecutionPolicy = seq>
inline auto v_k_nearest(const typename KDNode<T, Dim>::Ptr tree,
                        const py::EigenDRef<const Eigen::Matrix<T, Eigen::Dynamic, Dim>> queries,
                        std::size_t k) {
  std::vector<Eigen::Matrix<T, Eigen::Dynamic, Dim>> ret;
  ret.resize(static_cast<std::size_t>(queries.rows()));
  const auto fn = [&tree, k](const auto &query) {
    return k_nearest<T, Dim, ExecutionPolicy>(tree, query, k);
  };
  const auto &q_row = queries.rowwise();
  static const ExecutionPolicy policy{};
  std::transform(policy, q_row.begin(), q_row.end(), ret.begin(), fn);
  return ret;
}

template <typename T, std::size_t Dim, class ExecutionPolicy = seq>
inline auto k_nearest_periodic(const typename KDNode<T, Dim>::Ptr tree,
                               const py::EigenDRef<const Eigen::Vector<T, Dim>> query,
                               const py::EigenDRef<const Box<T, Dim>> box, const size_t k) {
  const auto buf = k_nearest_periodic_impl<T, Dim, ExecutionPolicy>(tree, query, box, k);
  Eigen::Matrix<T, Eigen::Dynamic, Dim> ret(buf.size(), Dim);
  const auto fn = [](const typename BufferTrait<T, Dim>::ElementType &elem) {
    return elem.first->point;
  };

  static const ExecutionPolicy policy{};
  std::transform(policy, buf.begin(), buf.end(), ret.rowwise().begin(), fn);

  return ret;
}

template <typename T, std::size_t Dim, class ExecutionPolicy = seq>
inline auto
v_k_nearest_periodic(const typename KDNode<T, Dim>::Ptr tree,
                     const py::EigenDRef<const Eigen::Matrix<T, Eigen::Dynamic, Dim>> queries,
                     const py::EigenDRef<const Box<T, Dim>> box, const size_t k) {
  std::vector<Eigen::Matrix<T, Eigen::Dynamic, Dim>> ret;
  ret.resize(static_cast<std::size_t>(queries.rows()));

  const auto fn = [&tree, &box, k](const auto &query) {
    return k_nearest_periodic<T, Dim, ExecutionPolicy>(tree, query, box, k);
  };

  const auto &q_row = queries.rowwise();
  static const ExecutionPolicy policy{};
  std::transform(policy, q_row.begin(), q_row.end(), ret.begin(), fn);

  return ret;
}

template <typename T, std::size_t Dim, class ExecutionPolicy = seq>
inline auto
clamp2periodic_box(const py::EigenDRef<const Eigen::Matrix<T, Eigen::Dynamic, Dim>> points,
                   const py::EigenDRef<const Box<T, Dim>> box) {
  Eigen::Matrix<T, Eigen::Dynamic, Dim> ret(points.rows(), Dim);
  const auto fn = [&box](const Eigen::Vector<T, Dim> &point) {
    return clamp2periodic_box_impl<T, Dim>(point, box);
  };

  static const ExecutionPolicy policy{};
  std::transform(policy, points.rowwise().begin(), points.rowwise().end(), ret.rowwise().begin(),
                 fn);

  return ret;
}

template <typename T, std::size_t Dim, class ExecutionPolicy = seq>
inline auto clamp2nearest(const py::EigenDRef<const Eigen::Matrix<T, Eigen::Dynamic, Dim>> points,
                          const py::EigenDRef<const Eigen::Vector<T, Dim>> query,
                          const py::EigenDRef<const Box<T, Dim>> box) {
  Eigen::Matrix<T, Eigen::Dynamic, Dim> ret(points.rows(), Dim);
  const auto fn = [&query, &box](const Eigen::Vector<T, Dim> &point) {
    return clamp2nearest_impl<T, Dim>(point, query, box);
  };

  static const ExecutionPolicy policy{};
  std::transform(policy, points.rowwise().begin(), points.rowwise().end(), ret.rowwise().begin(),
                 fn);
  return ret;
}

template <typename T, std::size_t Dim, class ExecutionPolicy = seq>
inline auto v_clamp2nearest(const std::vector<Eigen::Matrix<T, Eigen::Dynamic, Dim>> &points_list,
                            const py::EigenDRef<const Eigen::Vector<T, Dim>> query,
                            const py::EigenDRef<const Box<T, Dim>> box) {
  std::vector<Eigen::Matrix<T, Eigen::Dynamic, Dim>> ret{
      points_list.size(), Eigen::Matrix<T, Eigen::Dynamic, Dim>(points_list[0].rows(), Dim)};
  const auto fn = [&query, &box](const auto &points) {
    return clamp2nearest<T, Dim, ExecutionPolicy>(points, query, box);
  };

  static const ExecutionPolicy policy{};
  std::transform(policy, points_list.begin(), points_list.end(), ret.begin(), fn);

  return ret;
}
} // namespace arbol_kd

static constexpr std::size_t MAX_DEPTH = 3;
using Tree3df64 = arbol_kd::KDNode<arbol_kd::f64, 3>;
using Tree2df64 = arbol_kd::KDNode<arbol_kd::f64, 2>;

PYBIND11_MODULE(arbol_kd, m) {
  m.doc() = "arbol_kd module"; // Optional module docstring
  // Define the 2D KD-tree class
  py::class_<Tree2df64, Tree2df64::Ptr>(m, "KDNode2D")
      .def(py::init<>())
      .def(
          "__repr__",
          [](const Tree2df64 &self, std::size_t max_depth) {
            return arbol_kd::repr(&self, "", max_depth);
          },
          py::arg("max_depth") = MAX_DEPTH);
  m.def("build_tree_2d", &arbol_kd::build_tree<arbol_kd::f64, 2>, py::arg("points"),
        py::return_value_policy::move);
  m.def("k_nearest", &arbol_kd::k_nearest<arbol_kd::f64, 2, arbol_kd::par_unseq>, py::arg("tree"),
        py::arg("query"), py::arg("k"), py::return_value_policy::move);
  m.def("v_k_nearest", &arbol_kd::v_k_nearest<arbol_kd::f64, 2, arbol_kd::par_unseq>,
        py::arg("tree"), py::arg("queries"), py::arg("k"), py::return_value_policy::move);
  // Periodic version
  m.def("k_nearest_periodic", &arbol_kd::k_nearest_periodic<arbol_kd::f64, 2, arbol_kd::par_unseq>,
        py::arg("tree"), py::arg("query"), py::arg("box"), py::arg("k"),
        py::return_value_policy::move);
  m.def("v_k_nearest_periodic",
        &arbol_kd::v_k_nearest_periodic<arbol_kd::f64, 2, arbol_kd::par_unseq>, py::arg("tree"),
        py::arg("queries"), py::arg("box"), py::arg("k"), py::return_value_policy::move);
  m.def("clamp2periodic_box", &arbol_kd::clamp2periodic_box<arbol_kd::f64, 2, arbol_kd::par_unseq>,
        py::arg("points"), py::arg("box"), py::return_value_policy::move);
  m.def("clamp2nearest", &arbol_kd::clamp2nearest<arbol_kd::f64, 2, arbol_kd::par_unseq>,
        py::arg("points"), py::arg("query"), py::arg("box"), py::return_value_policy::move);
  m.def("v_clamp2nearest", &arbol_kd::v_clamp2nearest<arbol_kd::f64, 2, arbol_kd::par_unseq>,
        py::arg("points_list"), py::arg("queries"), py::arg("box"), py::return_value_policy::move);

  // Define the 3D KD-tree class
  py::class_<Tree3df64, Tree3df64::Ptr>(m, "KDNode3D")
      .def(py::init<>())
      .def(
          "__repr__",
          [](const Tree3df64 &self, std::size_t max_depth) {
            return arbol_kd::repr(&self, "", max_depth);
          },
          py::arg("max_depth") = MAX_DEPTH);
  m.def("build_tree_3d", &arbol_kd::build_tree<arbol_kd::f64, 3>, py::arg("points"),
        py::return_value_policy::move);
  m.def("k_nearest", &arbol_kd::k_nearest<arbol_kd::f64, 3, arbol_kd::par_unseq>, py::arg("tree"),
        py::arg("query"), py::arg("k"), py::return_value_policy::move);
  m.def("v_k_nearest", &arbol_kd::v_k_nearest<arbol_kd::f64, 3, arbol_kd::par_unseq>,
        py::arg("tree"), py::arg("queries"), py::arg("k"), py::return_value_policy::move);
  // Periodic version
  m.def("k_nearest_periodic", &arbol_kd::k_nearest_periodic<arbol_kd::f64, 3, arbol_kd::par_unseq>,
        py::arg("tree"), py::arg("query"), py::arg("box"), py::arg("k"),
        py::return_value_policy::move);
  m.def("v_k_nearest_periodic",
        &arbol_kd::v_k_nearest_periodic<arbol_kd::f64, 3, arbol_kd::par_unseq>, py::arg("tree"),
        py::arg("queries"), py::arg("box"), py::arg("k"), py::return_value_policy::move);
  m.def("clamp2periodic_box", &arbol_kd::clamp2periodic_box<arbol_kd::f64, 3, arbol_kd::par_unseq>,
        py::arg("points"), py::arg("box"), py::return_value_policy::move);
  m.def("clamp2nearest", &arbol_kd::clamp2nearest<arbol_kd::f64, 3, arbol_kd::par_unseq>,
        py::arg("points"), py::arg("query"), py::arg("box"), py::return_value_policy::move);
  m.def("v_clamp2nearest", &arbol_kd::v_clamp2nearest<arbol_kd::f64, 3, arbol_kd::par_unseq>,
        py::arg("points_list"), py::arg("queries"), py::arg("box"), py::return_value_policy::move);
}
