#include <pybind11/pybind11.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_autograd, m) {
  m.def(
      "backward",
      [](Graph g, bool retainGraph) {
        py::gil_scoped_release release;
        backward(g, retainGraph);
      },
      "g"_a,
      "retain_graph"_a = false);
  m.def(
      "backward",
      [](Graph g, const Graph& grad, bool retainGraph) {
        py::gil_scoped_release release;
        backward(g, grad, retainGraph);
      },
      "g"_a,
      "grad"_a,
      "retain_graph"_a = false);
}
