/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>

#include "parallel_compose.h"
#include "prefix_scan.h"

namespace gtn {
namespace detail {
namespace dataparallel {

// Convert from AOS to SOA
// Assumption: nodes are numbered [0..graph.numNodes())
//           : arcs are numbered [0..graph.numArcs())
GraphDataParallel convertToDataParallel(const Graph& graph) {
  GraphDataParallel graphDP;

  assert(graphDP.accept.empty());
  assert(graphDP.start.empty());

  // Safe since sizes are asserted to 0
  graphDP.accept.resize(graph.numNodes(), false);
  graphDP.start.resize(graph.numNodes(), false);

  graphDP.inArcOffset.resize(graph.numNodes());
  graphDP.outArcOffset.resize(graph.numNodes());

  graphDP.inArcs.resize(graph.numArcs());
  graphDP.outArcs.resize(graph.numArcs());

  graphDP.ilabels.resize(graph.numArcs());
  graphDP.olabels.resize(graph.numArcs());
  graphDP.srcNodes.resize(graph.numArcs());
  graphDP.dstNodes.resize(graph.numArcs());
  graphDP.weights.resize(graph.numArcs());

  for (auto i : graph.accept()) {
    assert(i >= 0);
    assert(i < graph.numNodes());
    graphDP.accept[i] = true;
  }

  for (auto i : graph.start()) {
    assert(i >= 0);
    assert(i < graph.numNodes());
    graphDP.start[i] = true;
  }

  for (int i = 0; i < graph.numNodes(); ++i) {
    graphDP.inArcOffset[i] = graph.numIn(i);
    graphDP.outArcOffset[i] = graph.numOut(i);
  }

  // Scan of offsets
  const int totalInArcs = prefixSumScan(graphDP.inArcOffset, false);
  const int totalOutArcs = prefixSumScan(graphDP.outArcOffset, false);
  assert(totalInArcs == totalOutArcs);
  assert(totalInArcs == graph.numArcs());

  for (int i = 0; i < graph.numNodes(); ++i) {
    int offset = graphDP.outArcOffset[i];

    for (auto j : graph.out(i)) {
      assert(j >= 0);
      assert(j < graph.numArcs());
      graphDP.outArcs[offset] = j;
      offset++;

      graphDP.ilabels[j] = graph.ilabel(j);
      graphDP.olabels[j] = graph.olabel(j);
      graphDP.srcNodes[j] = graph.srcNode(j);
      graphDP.dstNodes[j] = graph.dstNode(j);
      graphDP.weights[j] = graph.weight(j);
    }
  }

  for (int i = 0; i < graph.numNodes(); ++i) {
    int offset = graphDP.inArcOffset[i];

    for (auto j : graph.in(i)) {
      assert(j >= 0);
      assert(j < graph.numArcs());
      graphDP.inArcs[offset] = j;
      offset++;
    }
  }

  return graphDP;
}

// Convert from SOA to AOS
// The Graph is supposed to have no nodes and arcs and only supposed to have
// inputs set
// Assumption: nodes are numbered [0..graph.numNodes())
//           : arcs are numbered [0..graph.numArcs())
Graph convertFromDataParallel(const GraphDataParallel& graphDP) {
  assert(graphDP.inArcOffset.size() == graphDP.outArcOffset.size());
  assert(graphDP.inArcs.size() == graphDP.outArcs.size());
  assert(graphDP.inArcs.size() == graphDP.ilabels.size());
  assert(graphDP.ilabels.size() == graphDP.olabels.size());
  assert(graphDP.ilabels.size() == graphDP.srcNodes.size());
  assert(graphDP.ilabels.size() == graphDP.dstNodes.size());
  assert(graphDP.ilabels.size() == graphDP.weights.size());

  const size_t numNodes = graphDP.inArcOffset.size();
  const size_t numArcs = graphDP.inArcs.size();

  Graph graph;
  for (size_t i = 0; i < numNodes; ++i) {
    const int node = graph.addNode(graphDP.start[i], graphDP.accept[i]);
    assert(node == i);
  }

  for (size_t i = 0; i < numNodes; ++i) {
    const int start = graphDP.outArcOffset[i];
    const int end =
        (i == (numNodes - 1)) ? numArcs : graphDP.outArcOffset[i + 1];

    for (int j = start; j < end; ++j) {
      const int dstNode = graphDP.dstNodes[graphDP.outArcs[j]];
      const int ilabel = graphDP.ilabels[graphDP.outArcs[j]];
      const int olabel = graphDP.olabels[graphDP.outArcs[j]];
      const float weight = graphDP.weights[graphDP.outArcs[j]];

      auto newarc = graph.addArc(i, dstNode, ilabel, olabel, weight);
    }
  }
  return graph;
}

} // namespace dataparallel
} // namespace detail
} // namespace gtn
