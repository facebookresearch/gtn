/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
  
  m.def("save", py::overload_cast<const std::string&, const Graph&>(&save), "file_name"_a, "graph"_a);
  
  m.def("savetxt", py::overload_cast<const std::string&, const Graph&>(&saveTxt), "file_name"_a, "graph"_a);
  
  m.def("loadtxt", py::overload_cast<const std::string&>(&loadTxt), "file_name"_a);

}
