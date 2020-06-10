#pragma once

#include "graph.h"

namespace gtn {

// Compute the gradients of any inputs w.r.t graph
void backward(Graph graph);
} // namespace gtn
