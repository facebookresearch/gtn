cmake_minimum_required(VERSION 3.5.1)
project(gtn-experimental)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

function(build_experimental SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  target_sources(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/prefix_scan.cpp
  )
  target_sources(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/converters.cpp
  )
  target_sources(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/parallel_compose.cpp
  )
  target_link_libraries(
    ${target}
    PRIVATE
    gtn
  )
  target_include_directories(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/..
  )
endfunction(build_experimental)

build_experimental(test_parallel_compose.cpp)
