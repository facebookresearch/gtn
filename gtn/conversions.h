#pragma once

#include "graph.h"

namespace gtn {

/**
 * Create a linear chain graph with M+1 nodes and N edges between each node
 * from an array. The array is assumed to be in "edge-major" order, e.g. the
 * weights of all the outgoing edges for a node are assumed to be consecutive.
 * The labels of the edges between each node are the integers [0, ..., N-1].
 */
Graph arrayToLinearGraph(const float* src, int M, int N, bool calcGrad = true);

/**
 * Extract an array from a linear chain graph. The array should have space for
 * `g.numArcs()` elements. The weights stored in the array will be in
 * "edge-major" order, e.g. the weights of all the outgoing edges for a node
 * are assumed to be consecutive.
 */
void linearGraphToArray(Graph g, float* dst);

} // namespace gtn
