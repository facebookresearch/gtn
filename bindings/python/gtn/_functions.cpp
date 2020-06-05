#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_functions, m) {
  m.def("compose", compose, "a"_a, "b"_a);
  m.def("forward", forward, "a"_a);
  m.def("subtract", subtract, "a"_a, "b"_a);
  m.def("sum", sum, "a"_a);
}