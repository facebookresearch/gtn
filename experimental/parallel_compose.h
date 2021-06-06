/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "gtn/graph.h"

namespace gtn {
namespace detail {

namespace dataparallel {

// Change AOS to SOA
struct GraphDataParallel {
  // True if a node is accept or start, false otherwise
  std::vector<int> accept;
  std::vector<int> start;

  // One value per node - i-th value corresponds to i-th node
  // Last element is the total number of arcs, so that
  // each element and its neighbor forms a range
  std::vector<int> inArcOffset;
  std::vector<int> outArcOffset;

  // One value per arc
  std::vector<int> inArcs;
  std::vector<int> outArcs;

  // One value per arc
  // i-th value corresponds to i-th arc
  std::vector<int> ilabels;
  std::vector<int> olabels;
  std::vector<int> srcNodes;
  std::vector<int> dstNodes;
  std::vector<float> weights;
};

GraphDataParallel convertToDataParallel(const Graph& graph);

Graph convertFromDataParallel(const GraphDataParallel& graphDP);

Graph compose(const Graph& first, const Graph& second);

} // namespace dataparallel

} // namespace detail
} // namespace gtn
