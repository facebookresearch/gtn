#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_utils, m) {
  m.def("equal", equal, "g1"_a, "g2"_a);
  m.def("isomorphic", isomorphic, "g1"_a, "g2"_a);

  m.def(
      "write_dot",
      [](Graph g,
         const std::string& filename,
         const SymbolMap& isymbols = SymbolMap(),
         const SymbolMap& osymbols = SymbolMap()) {
        draw(g, filename, isymbols, osymbols);
      },
      "g"_a,
      "file_name"_a,
      "isymbols"_a = SymbolMap(),
      "osymbols"_a = SymbolMap());

  m.def("load", py::overload_cast<const std::string&>(&load), "file_name"_a);
}
