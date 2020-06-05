#include <pybind11/pybind11.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;

PYBIND11_MODULE(_graph, m) {
    py::class_<Arc>(m, "Arc")
        .def(py::init<Node* , Node* , int , int , float>())
        .def("up_node", &Arc::upNode)
        .def("down_node", &Arc::downNode)
        .def("ilabel", &Arc::ilabel)
        .def("olabel", &Arc::olabel)
        .def("label", &Arc::label)
        .def("weight", &Arc::weight)
        .def("set_weight", &Arc::setWeight)
        .def("grad", &Arc::grad)
        .def("add_grad", &Arc::addGrad)
        .def("zero_grad", &Arc::zeroGrad);
    
    py::class_<Node>(m, "Node")
        .def(py::init<int, bool, bool>())
        .def("add_in_arc", &Node::addInArc)
        .def("down_node", &Node::addOutArc)
        .def("num_in", &Node::numIn)
        .def("num_out", &Node::numOut)
        .def("index", &Node::index)
        .def("start", &Node::start)
        .def("accept", &Node::accept)
        .def("in", &Node::in)
        .def("out", &Node::out)
        .def("set_accept", &Node::setAccept);
}
