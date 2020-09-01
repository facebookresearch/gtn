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
      [](const std::function<Graph(const Graph&)>& function,
         const std::vector<Graph>& g1) {
        py::gil_scoped_release release;
        return parallelMap(function, g1);
      });

  m.def(
      "parallel_map",
      [](const std::function<Graph(const Graph&, const Graph&)>& function,
         const std::vector<Graph>& g1,
         const std::vector<Graph>& g2) {
        py::gil_scoped_release release;
        return parallelMap(function, g1, g2);
      });

  m.def(
      "parallel_map",
      [](const std::function<Graph(const std::vector<Graph>&)>& function,
         const std::vector<std::vector<Graph>>& graphList) {
        py::gil_scoped_release release;
        return parallelMap(function, graphList);
      });

  m.def(
      "parallel_map",
      [](const std::function<void(const Graph&, bool)>& function,
         const std::vector<Graph>& graphs,
         std::vector<bool>& bools) {
        py::gil_scoped_release release;
        return parallelMap(function, graphs, bools);
      });

  m.def(
      "parallel_map",
      [](const std::function<void(const Graph&, const Graph&, bool)>& function,
         const std::vector<Graph>& graphs1,
         const std::vector<Graph>& graphs2,
         std::vector<bool>& bools) {
        py::gil_scoped_release release;
        parallelMap(function, graphs1, graphs2, bools);
      });

  m.def(
      "parallel_map",
      [](const std::function<void(int)>& function,
         const std::vector<int>& ints) {
        py::gil_scoped_release release;
        parallelMap(function, ints);
      });
}
