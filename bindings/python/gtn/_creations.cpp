#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_creations, m) {
  m.def("scalar_graph", scalarGraph, "weight"_a, "calc_grad"_a = true);

  m.def(
      "linear_graph",
      [](int M, int N, bool calcGrad) { return linearGraph(M, N, calcGrad); },
      "M"_a,
      "N"_a,
      "calc_grad"_a = true);
}
