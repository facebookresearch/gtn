cmake_minimum_required(VERSION 3.5.1)

project(gtn-python)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH "${GTN_SOURCE_DIR}/cmake")

include(${CMAKE_MODULE_PATH}/Buildpybind11.cmake)
include(${CMAKE_MODULE_PATH}/pybind11Tools.cmake)

function (add_pybind11_extension ext_name)
  pybind11_add_module(
    ${ext_name}
    ${CMAKE_CURRENT_SOURCE_DIR}/gtn/${ext_name}.cpp
    )
  
  target_include_directories(
    ${ext_name} 
    PUBLIC 
    ${GTN_SOURCE_DIR}
    )

  target_link_libraries(
    ${ext_name}
    PRIVATE
    gtn
    )

  add_dependencies(${ext_name} pybind11)
endfunction ()

add_pybind11_extension(_graph)