#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_functions, m) {
  m.def(
      "add",
      [](const Graph& g1, const Graph& g2) {
        py::gil_scoped_release release;
        return add(g1, g2);
      },
      "g1"_a,
      "g2"_a);
  m.def(
      "add",
      [](const std::vector<Graph>& graphs1, const std::vector<Graph>& graphs2) {
        py::gil_scoped_release release;
        return parallelMap(add, graphs1, graphs2);
      },
      "graphs1"_a,
      "graphs2"_a);
  m.def(
      "concat",
      [](const Graph& g1, const Graph& g2) {
        py::gil_scoped_release release;
        return concat(g1, g2);
      },
      "g1"_a,
      "g2"_a);
  m.def(
      "concat",
      [](const std::vector<Graph>& graphs1, const std::vector<Graph>& graphs2) {
        py::gil_scoped_release release;
        return parallelMap(
            static_cast<Graph (*)(const Graph&, const Graph&)>(&concat),
            graphs1,
            graphs2);
      },
      "graphs1"_a,
      "graphs2"_a);
  m.def(
      "concat",
      [](const std::vector<Graph>& graphs) {
        py::gil_scoped_release release;
        return concat(graphs);
      },
      "graphs"_a);
  m.def(
      "concat",
      [](const std::vector<std::vector<Graph>>& graphs) {
        py::gil_scoped_release release;
        return parallelMap(
            static_cast<Graph (*)(const std::vector<Graph>&)>(&concat), graphs);
      },
      "graphs"_a);
  m.def(
      "intersect",
      [](const Graph& g1, const Graph& g2) {
        py::gil_scoped_release release;
        return intersect(g1, g2);
      },
      "g1"_a,
      "g2"_a);
  m.def(
      "intersect",
      [](const std::vector<Graph>& graphs1, const std::vector<Graph>& graphs2) {
        py::gil_scoped_release release;
        return parallelMap(intersect, graphs1, graphs2);
      },
      "graphs1"_a,
      "graphs2"_a);
  m.def(
      "compose",
      [](const Graph& g1, const Graph& g2) {
        py::gil_scoped_release release;
        return compose(g1, g2);
      },
      "g1"_a,
      "g2"_a);
  m.def(
      "compose",
      [](const std::vector<Graph>& graphs1, const std::vector<Graph>& graphs2) {
        py::gil_scoped_release release;
        return parallelMap(compose, graphs1, graphs2);
      },
      "graphs1"_a,
      "graphs2"_a);
  m.def(
      "closure",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return closure(g);
      },
      "g"_a);
  m.def(
      "closure",
      [](const std::vector<Graph>& graphs) {
        py::gil_scoped_release release;
        return parallelMap(closure, graphs);
      },
      "graphs"_a);
  m.def(
      "forward_score",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return forwardScore(g);
      },
      "g"_a);
  m.def(
      "forward_score",
      [](const std::vector<Graph>& graphs) {
        py::gil_scoped_release release;
        return parallelMap(forwardScore, graphs);
      },
      "graphs"_a);
  m.def(
      "negate",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return negate(g);
      },
      "g"_a);
  m.def(
      "negate",
      [](const std::vector<Graph>& graphs) {
        py::gil_scoped_release release;
        return parallelMap(negate, graphs);
      },
      "graphs"_a);
  m.def(
      "project_input",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return projectInput(g);
      },
      "g"_a);
  m.def(
      "project_input",
      [](const std::vector<Graph>& graphs) {
        py::gil_scoped_release release;
        return parallelMap(projectInput, graphs);
      },
      "graphs"_a);
  m.def(
      "project_output",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return projectOutput(g);
      },
      "g"_a);
  m.def(
      "project_output",
      [](const std::vector<Graph>& graphs) {
        py::gil_scoped_release release;
        return parallelMap(projectOutput, graphs);
      },
      "graphs"_a);
  m.def(
      "remove",
      [](const Graph& g, int label) {
        py::gil_scoped_release release;
        return remove(g, label);
      },
      "g"_a,
      "label"_a = Graph::epsilon);
  m.def(
      "remove",
      [](const std::vector<Graph>& graphs, const std::vector<int>& labels) {
        py::gil_scoped_release release;
        return parallelMap(
            static_cast<Graph (*)(const Graph&, int)>(&remove), graphs, labels);
      },
      "graphs"_a,
      "labels"_a = std::vector<int>{ Graph::epsilon });
  m.def(
      "subtract",
      [](const Graph& g1, const Graph& g2) {
        py::gil_scoped_release release;
        return subtract(g1, g2);
      },
      "g1"_a,
      "g2"_a);
  m.def(
      "subtract",
      [](const std::vector<Graph>& graphs1, const std::vector<Graph>& graphs2) {
        py::gil_scoped_release release;
        return parallelMap(subtract, graphs1, graphs2);
      },
      "graphs1"_a,
      "graphs2"_a);
  m.def(
      "union",
      [](const std::vector<Graph>& graphs) {
        py::gil_scoped_release release;
        return union_(graphs);
      },
      "graphs"_a);
  m.def(
      "union",
      [](const std::vector<std::vector<Graph>>& graphs) {
        py::gil_scoped_release release;
        return parallelMap(union_, graphs);
      },
      "graphs"_a);
  m.def(
      "viterbi_score",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return viterbiScore(g);
      },
      "g"_a);
  m.def(
      "viterbi_score",
      [](const std::vector<Graph>& graphs) {
        py::gil_scoped_release release;
        return parallelMap(viterbiScore, graphs);
      },
      "graphs"_a);
  m.def(
      "viterbi_path",
      [](const Graph& g) {
        py::gil_scoped_release release;
        return viterbiPath(g);
      },
      "g"_a);
  m.def(
      "viterbi_path",
      [](const std::vector<Graph>& graphs) {
        py::gil_scoped_release release;
        return parallelMap(viterbiPath, graphs);
      },
      "graphs"_a);
}
