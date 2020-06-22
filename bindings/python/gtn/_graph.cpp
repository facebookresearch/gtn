#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_graph, m) {
  py::class_<Graph>(m, "Graph")
      .def(py::init<bool>(), "calc_grad"_a = true)
      .def(
          "add_node",
          py::overload_cast<bool, bool>(&Graph::addNode),
          "start"_a = false,
          "accept"_a = false)
      .def(
          "add_arc",
          py::overload_cast<int, int, int>(&Graph::addArc))
      .def(
          "add_arc",
          py::overload_cast<int, int, int, int, float>(&Graph::addArc))
      .def("num_arcs", &Graph::numArcs)
      .def("num_nodes", &Graph::numNodes)
      .def("num_start", &Graph::numStart)
      .def("num_accept", &Graph::numAccept)
      .def("acceptor", &Graph::acceptor)
      .def("item", &Graph::item)
      .def("calc_grad", &Graph::calcGrad)
      .def("zero_grad", &Graph::zeroGrad)
      .def("__repr__", [](const Graph& a) {
        std::ostringstream ss;
        print(a, ss);
        return ss.str();
      });
}
