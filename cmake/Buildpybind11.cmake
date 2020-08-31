include(ExternalProject)

if (GTN_BUILD_PYTHON_BINDINGS_NO_GIL)
  message(STATUS "Building pybind11 with no GIL")
  # GIL-free pybind11
  set(pybind11_URL https://github.com/colesbury/pybind11.git)
  set(pybind11_TAG f4e81bfc95e57eb6b4c32698ab5308903c3c2a46)
else ()
  # pybind11 source
  set(pybind11_URL https://github.com/pybind/pybind11.git)
  set(pybind11_TAG eeb1044818af5b70761deae602c49eba439164dc)
endif ()

# Download pybind11
ExternalProject_Add(
  pybind11
  PREFIX pybind11
  GIT_REPOSITORY ${pybind11_URL}
  GIT_TAG ${pybind11_TAG}
  BUILD_IN_SOURCE 0
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

ExternalProject_Get_Property(pybind11 SOURCE_DIR)
set(pybind11_INCLUDE_DIR "${SOURCE_DIR}/include")
