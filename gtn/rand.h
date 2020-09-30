/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gtn/graph.h"

namespace gtn {

/** @ingroup functions
 * Attempt to sample an accepting path from the graph. If the graph has
 * dead-ends or does accept any paths, then an empty path may be returned. Note
 * that the empty path "{}" is different from the path which is the empty
 * string "{ε}".
 * @param g The graph to sample from.
 * @param maxLength The maximum length of a sampled path.
 */
Graph sample(const Graph& g, size_t maxLength = 1000);

/** @ingroup comparisons
 * Compare two graphs by sampling paths and checking they have the same scores
 * in both graphs.
 * @param g1 A graph to be compared.
 * @param g2 A graph to be compared.
 * @param numSamples The number of samples to use. The more samples the more
 * likely the result is accurate.
 * @param tol The largest allowed absolute difference between the path score
   from each graph.
 * @param maxLength The maximum length of sampled paths.
 */
bool randEquivalent(
    const Graph& g1,
    const Graph& g2,
    size_t numSamples = 100,
    double tol = 1e-4,
    size_t maxLength = 1000);

} // namespace gtn
