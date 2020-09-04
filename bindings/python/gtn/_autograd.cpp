#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
      [](std::vector<Graph> graphs, const std::vector<int>& retainGraphs) {
        py::gil_scoped_release release;
        parallelMap(
            static_cast<void (*)(Graph, bool)>(&backward),
            graphs,
            retainGraphs);
      },
      "graphs"_a,
      "retain_graphs"_a = std::vector<int>({0}));

  m.def(
      "backward",
      [](Graph g, const Graph& grad, bool retainGraph) {
        py::gil_scoped_release release;
        backward(g, grad, retainGraph);
      },
      "g"_a,
      "grad"_a,
      "retain_graph"_a = false);
  m.def(
      "backward",
      [](std::vector<Graph> graphs,
         const std::vector<Graph>& grads,
         const std::vector<int>& retainGraphs) {
        py::gil_scoped_release release;
        parallelMap(
            static_cast<void (*)(Graph, const Graph&, bool)>(&backward),
            graphs,
            grads,
            retainGraphs);
      },
      "graphs"_a,
      "grads"_a,
      "retain_graphs"_a = std::vector<int>({0}));
}
