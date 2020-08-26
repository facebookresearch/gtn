#pragma once

#include "gtn/graph.h"

namespace gtn {

/**
 * Compute the gradients of any inputs w.r.t `graph` using the chain rule to
 * incorporate `grad`, a gradient of another function with respect to `graph`.
 * If `retainGraph = false` (default) then the autograd graph will not be saved
 * and un-referenced nodes in the autograd graph may be destroyed.
 *
 * @param graph The graph to compute gradients with respect to.
 * @param grad A seed gradient, typically set to be a gradient of another
 *   function with respect to graph.
 * @param retainGraph Whether or not to save the autograd graph. Setting this
 *   to False is more memory efficient as temporary Graphs created during the
 *   forward computation may be destroyed.
 */
void backward(Graph graph, const Graph& grad, bool retainGraph = false);

/**
 * Compute the gradients of any inputs w.r.t `graph`. If `retainGraph = false`
 * (default) then the autograd graph will not be saved and un-referenced nodes
 * in the autograd graph may be destroyed.
 * \verbatim embed:rst:leading-asterisk
 * See the overload :cpp:func:`gtn::backward`.
 * \endverbatim
 */
void backward(Graph graph, bool retainGraph = false);

} // namespace gtn
