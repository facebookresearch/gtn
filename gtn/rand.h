#pragma once

#include "graph.h"

namespace gtn {

/* Attempt to sample a successful path from the graph. If the graph has
 * dead-ends or does not have any accepting paths, then an empty path may be
 * returned. Note that there is a difference between an empty path "{}" and the
 * path which is the empty string "{ε}". */
Graph sample(Graph graph, size_t maxLength = 1000);

/* Compare two graphs by sampling `numSamples` (using `sample`) and
 * checking that the score (over all accepting paths) for the sample is
 * within `tol` for the two graphs. */
bool randEquivalent(
    Graph a,
    Graph b,
    size_t numSamples = 100,
    double tol = 1e-4,
    size_t maxLength = 1000);

} // namespace gtn
