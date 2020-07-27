#pragma once

#include "graph.h"

namespace gtn {

/* Compute the gradients of any inputs w.r.t graph. If `retainGraph = false`
 * (default) then the autograd graph will not be saved and un-referenced nodes
 * in the autograd graph may be destroyed.
 */
void backward(Graph graph, bool retainGraph = false);

/* Compute the gradients of any inputs w.r.t graph using the chain rule to
 * incorporate `grad`, a gradient of another function with respect to `graph`.
 * If `retainGraph = false` (default) then the autograd graph will not be saved
 * and un-referenced nodes in the autograd graph may be destroyed.
 */
void backward(Graph graph, const Graph& grad, bool retainGraph = false);

} // namespace gtn
