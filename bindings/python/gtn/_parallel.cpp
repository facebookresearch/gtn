#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_parallel, m) {
  m.def(
      "parallel_for",
      [](const std::function<void(int)>& function,
         const std::vector<int>& ints) -> void {
        py::gil_scoped_release release;
        parallelMap(function, ints);
      });
}
