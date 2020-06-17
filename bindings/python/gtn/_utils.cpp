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
      "create_linear_graph",
      [](py::bytes input,
         int T,
         int C,
         bool calcGrad) {
        return createLinearGraph(
            castBytes<float*>(input),
            T,
            C,
            calcGrad);
      });
    
     m.def(
      "extract_linear_grad",
      [](Graph g,
         float scale,
         py::bytes grad) {
         extractLinearGrad(
          g,
          scale,
          castBytes<float*>(grad)
          );
      });
}
