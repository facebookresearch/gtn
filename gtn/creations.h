#pragma once

#include "gtn/graph.h"

namespace gtn {

Graph scalarGraph(float val, bool calcGrad = true);

/**
 * Create a linear chain graph with M + 1 nodes and N edges between each node.
 * The labels of the edges between each node are the integers [0, ..., N-1].
 */
Graph linearGraph(int M, int N, bool calcGrad = true);

} // namespace gtn
