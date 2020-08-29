#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_parallel, m) {
  m.def(
      "parallel_map",
      [](const std::function<Graph(Graph)>& function, std::vector<Graph>& g1) {
        py::gil_scoped_release release;
        return parallelMap(function, g1);
      });

  m.def(
      "parallel_map",
      [](const std::function<Graph(Graph, Graph)>& function,
         std::vector<Graph>& g1,
         std::vector<Graph>& g2) {
        py::gil_scoped_release release;
        return parallelMap(function, g1, g2);
      });

  m.def(
      "parallel_map",
      [](const std::function<Graph(std::vector<Graph>)>& function,
         std::vector<std::vector<Graph>>& graphList) {
        py::gil_scoped_release release;
        return parallelMap(function, graphList);
      });

  m.def(
      "parallel_map",
      [](const std::function<void(Graph, bool)>& function,
         std::vector<Graph>& graphs,
         // This accepts an int (but also bools) because of an issue with
         // binding non-const lvalue references of type bool& to bool rvalues
         std::vector<bool>& bools) {
        py::gil_scoped_release release;
        return parallelMap(function, graphs, bools);
      });

  m.def(
      "parallel_map",
      [](const std::function<void(Graph, Graph, bool)>& function,
         std::vector<Graph>& graphs1,
         std::vector<Graph>& graphs2,
         std::vector<bool>& bools) {
        py::gil_scoped_release release;
        parallelMap(function, graphs1, graphs2, bools);
      });
}
