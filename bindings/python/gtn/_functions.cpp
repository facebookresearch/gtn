#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_functions, m) {
  m.def("add",
      [](const Graph& g1, const Graph& g2) {
        py::gil_scoped_release release;
        return add(g1, g2);
      },
      "g1"_a, "g2"_a);
  m.def(
      "concat",
      [](const Graph& g1, const Graph& g2) {
        py::gil_scoped_release release;
        return concat(g1, g2);
      },
      "g1"_a, "g2"_a);
  m.def(
      "concat",
      [](const std::vector<Graph>& graphs) {
        py::gil_scoped_release release;
        return concat(graphs);
      },
      "graphs"_a);
  m.def("intersect",
      [](const Graph& g1, const Graph& g2) {
        py::gil_scoped_release release;
        return intersect(g1, g2);
      },
      "g1"_a, "g2"_a);
  m.def("compose",
      [](const Graph& g1, const Graph& g2) {
        py::gil_scoped_release release;
        return compose(g1, g2);
      },
      "g1"_a, "g2"_a);
  m.def("closure",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return closure(g);
      },
      "g"_a);
  m.def("forward_score",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return forwardScore(g);
      },
      "g"_a);
  m.def("negate",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return negate(g);
      },
      "g"_a);
  m.def("project_input",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return projectInput(g);
      },
      "g"_a);
  m.def("project_output",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return projectOutput(g);
      },
      "g"_a);
  m.def(
      "remove",
      [](const Graph& g, int label) {
        py::gil_scoped_release release;
        return remove(g, label);
      },
      "g"_a,
      "label"_a = Graph::epsilon);
  m.def("subtract",
      [](const Graph& g1, const Graph& g2) {
        py::gil_scoped_release release;
        return subtract(g1, g2);
      },
      "g1"_a, "g2"_a);
  m.def("union",
      [](const std::vector<Graph>& graphs) {
        py::gil_scoped_release release;
        return union_(graphs);
      },
      "graphs"_a);
  m.def("viterbi_score",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return viterbiScore(g);
      },
      "g"_a);
  m.def("viterbi_path",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return viterbiPath(g);
      },
      "g"_a);
}
