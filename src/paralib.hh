/*
 * Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Wrapper header for parallel algorithms.
 */

#ifndef ARBOL_KD_PARALIB_HH
#define ARBOL_KD_PARALIB_HH

#if defined(ARBOL_KD_USE_ONEDPL)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#elif defined(ARBOL_KD_USE_STD)
#include <algorithm>
#include <execution>
#else
#error                                                                                             \
    "No parallel execution policy defined. Please define ARBOL_KD_USE_ONEDPL or ARBOL_KD_USE_STD."
#endif

namespace arbol_kd {
using par_unseq = std::execution::parallel_unsequenced_policy;
using par = std::execution::parallel_policy;
using seq = std::execution::sequenced_policy;
} // namespace arbol_kd

#endif // ARBOL_KD_PARALIB_HH
