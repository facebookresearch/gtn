#include <pybind11/pybind11.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_autograd, m) {
  m.def("backward", backward, "graph"_a);
}