#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_functions, m) {
  m.def("add", add, "lhs"_a, "rhs"_a);
  m.def(
      "concat",
      py::overload_cast<const Graph&, const Graph&>(&concat),
      "lhs"_a,
      "rhs"_a);
  m.def(
      "concat",
      py::overload_cast<const std::vector<Graph>&>(&concat),
      "graphs"_a);
  m.def("intersect", intersect, "first"_a, "second"_a);
  m.def("compose", compose, "first"_a, "second"_a);
  m.def("closure", closure, "graph"_a);
  m.def("forward_score", forwardScore, "graph"_a);
  m.def("negate", negate, "input"_a);
  m.def("project_input", projectInput, "other"_a);
  m.def("project_output", projectOutput, "other"_a);
  m.def(
      "remove",
      py::overload_cast<const Graph&, int>(&remove),
      "other"_a,
      "label"_a = Graph::epsilon);
  m.def("subtract", subtract, "lhs"_a, "rhs"_a);
  m.def("union", union_, "graphs"_a);
  m.def("viterbi_score", viterbiScore, "graph"_a);
  m.def("viterbi_path", viterbiPath, "graph"_a);
}
