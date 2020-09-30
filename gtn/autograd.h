/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gtn/graph.h"

namespace gtn {

/**
 * Compute the gradients of any inputs w.r.t `g` using the chain rule to
 * incorporate `grad`, a gradient of another function with respect to `graph`.
 * If `retainGraph = false` (default) then the autograd graph will not be saved
 * and un-referenced nodes in the autograd graph may be destroyed.
 *
 * @param g The graph to compute gradients with respect to.
 * @param grad A seed gradient, typically set to be a gradient of another
 *   function with respect to graph.
 * @param retainGraph Whether or not to save the autograd graph. Setting this
 *   to False is more memory efficient as temporary Graphs created during the
 *   forward computation may be destroyed.
 */
void backward(Graph g, const Graph& grad, bool retainGraph = false);

/**
 * Compute the gradients of any inputs w.r.t `g`. If `retainGraph = false`
 * (default) then the autograd graph will not be saved and un-referenced nodes
 * in the autograd graph may be destroyed.
 * \verbatim embed:rst:leading-asterisk
 * See the overload :cpp:func:`gtn::backward`.
 * \endverbatim
 */
void backward(Graph g, bool retainGraph = false);

} // namespace gtn
