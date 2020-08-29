cmake_minimum_required(VERSION 3.5.1)

set(
  GTN_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/autograd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/autograd.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/functions/compose.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/functions/compose.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/creations.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/creations.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/parallel/parallel_map.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/parallel/parallel_map.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/functions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/functions.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/graph.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/gtn.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/rand.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/rand.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/functions/shortest.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/functions/shortest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/utils.h
)
