#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_utils, m) {
  m.def("equal", equal, "first"_a, "second"_a);
  m.def("isomorphic", isomorphic, "first"_a, "second"_a);

  m.def(
      "write_dot",
      [](Graph graph,
         const std::string& filename,
         const SymbolMap& isymbols = SymbolMap(),
         const SymbolMap& osymbols = SymbolMap()) {
        draw(graph, filename, isymbols, osymbols);
      },
      "graph"_a,
      "file_name"_a,
      "isymbols"_a = SymbolMap(),
      "osymbols"_a = SymbolMap());

  m.def("load", py::overload_cast<const std::string&>(&load), "file_name"_a);
}
