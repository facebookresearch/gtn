#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

template <class T>
static T castBytes(const py::bytes& b) {
  static_assert(
      std::is_standard_layout<T>::value,
      "types represented as bytes must be standard layout");
  std::string s = b;
  if (s.size() != sizeof(T)) {
    throw std::runtime_error("wrong py::bytes size to represent object");
  }
  return *reinterpret_cast<const T*>(s.data());
}

PYBIND11_MODULE(_utils, m) {
  m.def(
      "draw",
      [](Graph graph,
         const std::string& filename,
         const SymbolMap& isymbols = SymbolMap(),
         const SymbolMap& osymbols = SymbolMap()) {
        draw(graph, filename, isymbols, osymbols);
      });

   m.def(
      "array_to_linear_graph",
      [](py::bytes src,
         int M,
         int N,
         bool calcGrad) {
        return arrayToLinearGraph(
            castBytes<float*>(src),
            M,
            N,
            calcGrad);
      });

     m.def(
      "linear_graph_to_array",
      [](Graph g,
         float scale,
         py::bytes dst) {
         linearGraphToArray(
          g,
          castBytes<float*>(dst)
          );
      });
}
