# Copyright (c) 2025 Materials Modelling Lab, The University of Tokyo
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(arbol_kd LANGUAGES CXX DESCRIPTION "KD Tree library")
set(CMAKE_CXX_STANDARD 17)

# find_package(Python COMPONENTS Interpreter Development)
set(PYBIND11_FINDPYTHON on)
find_package(pybind11 3.0.0 CONFIG QUIET)
find_package(Eigen3 QUIET)
find_package(oneDPL QUIET)

if (NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v3.0.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

if (NOT Eigen3_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.1
  )
  FetchContent_MakeAvailable(eigen)
endif()

function(set_module_output module outdir)
  set_target_properties(${module} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${outdir})

  # windows needs a build type variant
  foreach(config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${config} config)
    set_target_properties(${module} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_${config} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${outdir})
  endforeach()
endfunction()

pybind11_add_module(arbol_kd)
set_module_output(arbol_kd ${PROJECT_NAME})
target_sources(arbol_kd PRIVATE ${PROJECT_SOURCE_DIR}/src/arbol_kd.cc)
target_link_libraries(arbol_kd PRIVATE Eigen3::Eigen)

if (oneDPL_FOUND)
  target_compile_definitions(arbol_kd PRIVATE ARBOL_KD_USE_ONEDPL)
  target_link_libraries(arbol_kd PRIVATE oneDPL)
else()
  message(STATUS "oneDPL not found, using std parallel algorithms with TBB backend")
  find_package(TBB QUIET)
  if (NOT TBB_FOUND)
    include(FetchContent)
    set(TBB_TEST OFF CACHE BOOL "Disable oneTBB tests when vendoring" FORCE)
    FetchContent_Declare(
      tbb
      GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
      GIT_TAG v2023.1.0
    )
    FetchContent_MakeAvailable(tbb)
  endif()
  target_compile_definitions(arbol_kd PRIVATE ARBOL_KD_USE_STD)
  target_link_libraries(arbol_kd PRIVATE TBB::tbb)
endif()
