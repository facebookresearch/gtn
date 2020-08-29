#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_functions, m) {
  m.def("add", add, "g1"_a, "g2"_a);
  m.def(
      "concat",
      py::overload_cast<const Graph&, const Graph&>(&concat),
      "g1"_a,
      "g2"_a);
  m.def(
      "concat",
      py::overload_cast<const std::vector<Graph>&>(&concat),
      "graphs"_a);
  m.def("intersect", intersect, "g1"_a, "g2"_a);
  m.def("compose", compose, "g1"_a, "g2"_a);
  m.def("closure", closure, "g"_a);
  m.def("forward_score", forwardScore, "g"_a);
  m.def("negate", negate, "input"_a);
  m.def("project_input", projectInput, "g"_a);
  m.def("project_output", projectOutput, "g"_a);
  m.def(
      "remove",
      py::overload_cast<const Graph&, int>(&remove),
      "g"_a,
      "label"_a = Graph::epsilon);
  m.def("subtract", subtract, "g1"_a, "g2"_a);
  m.def("union", union_, "graphs"_a);
  m.def("viterbi_score", viterbiScore, "g"_a);
  m.def("viterbi_path", viterbiPath, "g"_a);
}
