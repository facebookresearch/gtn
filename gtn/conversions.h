#pragma once

#include "graph.h"

namespace gtn {

/**
 * Create an emission graph from sequential input with `T` timesteps
 * and `C` channels (feature size)
 */
Graph createLinearGraph(float* input, int T, int C, bool calcGrad = true);

/**
 * Extract gradients from all the arcs in the graph in the same order
 * as they are inserted via `createLinearGraph`
 */
void extractLinearGrad(Graph g, float scale, float* grad);

} // namespace gtn
