cmake_minimum_required(VERSION 3.5.1)

project(gtn)

# ----------------------------- Setup -----------------------------
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------- Configuration -----------------------------
option(GTN_BUILD_TESTS "Build tests for gtn" ON)
option(GTN_BUILD_BENCHMARKS "Build benchmarks for gtn" ON)
option(GTN_BUILD_EXAMPLES "Build examples for gtn" ON)
option(GTN_BUILD_EXPERIMENTAL "Build experimental for gtn" ON)
option(GTN_BUILD_PYTHON_BINDINGS "Build python bindings for gtn" OFF)

# ----------------------------- Dependencies -----------------------------
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ----------------------------- Lib -----------------------------
# Core lib sources -- maybe make me a submodule eventually
include(${CMAKE_SOURCE_DIR}/gtn/CMakeLists.txt)

# Main lib
add_library(
  gtn
  ${GTN_SOURCES}
)
add_subdirectory(${PROJECT_SOURCE_DIR}/gtn)

target_link_libraries(gtn PUBLIC ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(
  gtn
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

# Python Bindings
if (GTN_BUILD_PYTHON_BINDINGS)
  message(STATUS "Building Python bindings.")
  include(${CMAKE_MODULE_PATH}/Buildpybind11.cmake)
  include(${CMAKE_MODULE_PATH}/pybind11Tools.cmake)
  add_subdirectory(${PROJECT_SOURCE_DIR}/bindings/python)
endif()

# Tests
if (GTN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(${PROJECT_SOURCE_DIR}/test)
endif()

# Benchmarks
if (GTN_BUILD_BENCHMARKS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/benchmarks)
endif()

# Examples
if (GTN_BUILD_EXAMPLES)
  add_subdirectory(${PROJECT_SOURCE_DIR}/examples)
endif()

# Experimental
if (GTN_BUILD_EXPERIMENTAL)
  add_subdirectory(${PROJECT_SOURCE_DIR}/experimental)
endif()


# ----------------------------- Installation -----------------------------
# Installation directories
set(GTN_INSTALL_INC_DIR "include" CACHE PATH "Install path for headers")
set(GTN_INSTALL_INC_DIR_HEADER_LOC ${GTN_INSTALL_INC_DIR}/gtn)
set(GTN_INSTALL_LIB_DIR "lib" CACHE PATH "Install path for libraries")
set(GTN_INSTALL_BIN_DIR "bin" CACHE PATH "Install path for binaries")
# Other assets
set(GTN_INSTALL_ASSETS_BASE_DIR "share/gtn")
set(GTN_INSTALL_CMAKE_DIR "${GTN_INSTALL_ASSETS_BASE_DIR}/cmake" CACHE PATH "Install path for CMake files")

# Main target
install(
  TARGETS gtn
  EXPORT gtnTargets
  COMPONENT gtn
  PUBLIC_HEADER DESTINATION gtn
  RUNTIME DESTINATION ${GTN_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${GTN_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${GTN_INSTALL_LIB_DIR}
  FRAMEWORK DESTINATION framework
  INCLUDES DESTINATION ${GTN_INSTALL_INC_DIR}
)

# Write and install targets file
install(
  EXPORT
  gtnTargets
  NAMESPACE
  gtn::
  DESTINATION
  ${GTN_INSTALL_CMAKE_DIR}
  COMPONENT
  cmake)

# Move headers
install(
  DIRECTORY
  ${CMAKE_SOURCE_DIR}/gtn/ # module headers in ./gtn
  COMPONENT
  headers
  DESTINATION
  ${GTN_INSTALL_INC_DIR_HEADER_LOC}
  FILES_MATCHING # preserve directory structure
  PATTERN  "*.h"
  )

# Write config file
include(CMakePackageConfigHelpers)
set(INCLUDE_DIRS include)
set(CMAKE_DIR ${GTN_INSTALL_CMAKE_DIR})
configure_package_config_file(
  ${CMAKE_MODULE_PATH}/gtnConfig.cmake.in
  cmake/install/${GTN_CONFIG_CMAKE_BUILD_DIR}/gtnConfig.cmake
  INSTALL_DESTINATION
  ${GTN_INSTALL_CMAKE_DIR}
  PATH_VARS INCLUDE_DIRS CMAKE_DIR
  )
install(FILES
  ${PROJECT_BINARY_DIR}/cmake/install/gtnConfig.cmake
  DESTINATION ${GTN_INSTALL_CMAKE_DIR}
  COMPONENT cmake
  )
