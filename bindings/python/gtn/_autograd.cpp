#include <pybind11/pybind11.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_autograd, m) {
  m.def(
      "backward",
      py::overload_cast<Graph, bool>(&backward),
      "g"_a,
      "retain_graph"_a = false);
  m.def(
      "backward",
      py::overload_cast<Graph, const Graph&, bool>(&backward),
      "g"_a,
      "grad"_a,
      "retain_graph"_a = false);
}
