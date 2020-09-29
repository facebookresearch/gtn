/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gtn/graph.h"

namespace gtn {

/**
 * \defgroup creations Creation Functions
 @{
 }@
 */

/**
 * Creates a scalar graph - a graph with a single arc between two nodes with a
 * given weight value and a `epsilon` label.
 * \ingroup creations
 */
Graph scalarGraph(float val, bool calcGrad = true);

/**
 * Create a linear chain graph with `M + 1` nodes and `N` edges between each node.
 * The labels of the edges between each node are the integers `[0, ..., N - 1]`.
 * \ingroup creations
 */
Graph linearGraph(int M, int N, bool calcGrad = true);

} // namespace gtn
