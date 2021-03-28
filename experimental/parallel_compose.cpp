/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <tuple>

#include "parallel_compose.h"

namespace gtn {
namespace detail {
namespace dataparallel {

namespace {
// Exclusive/Inclusive prefix sum. The returned vector
// has one more element
int prefixSumScan(std::vector<int>& input, bool appendSum) {
  int sum = 0;
  for (size_t i = 0; i < input.size(); ++i) {
    auto count = input[i];
    input[i] = sum;
    sum += count;
  }
  if (appendSum) {
    input.push_back(sum);
  }

  return sum;
}

// TODO: Duplicate - should be removed
inline int TwoDToOneDIndex(int n1, int n2, int n1Extent) {
  assert(n1 < n1Extent);
  return n1 + n2 * n1Extent;
}

inline std::pair<int, int> OneDToTwoDIndex(int n, int n1Extent) {
  assert(n1Extent > 0);
  const int n2 = n / n1Extent;
  const int n1 = n % n1Extent;
  return std::make_pair(n1, n2);
}

bool checkAnyTrue(const std::vector<bool>& flags) {
  for (auto i : flags) {
    if (i == true) {
      return i;
    }
  }
  return false;
}

// Map thread id to corresponding node and arc pair
// Also map thread id to two flags checkEpsilonArcPair.first,
// checkEpsilonArcPair.second When checkEpsilonArcPair.first is set,
// corresponding tid will check for arcs with epsilon arcs in the node from
// first graph Same logic happens for checkEpsilonArcPair.second Search to find
// which node pair this tid will fall into Linear search for now
// (arcCrossProductOffset is sorted by definition)
std::tuple<
    bool,
    std::pair<int, int>,
    std::pair<int, int>,
    bool,
    std::pair<bool, bool>>
computeNodeAndArcPair(
    int tid,
    const std::vector<int>& arcCrossProductOffset,
    const std::vector<std::pair<int, int>>& toExploreNumArcs,
    const std::vector<std::pair<int, int>>& toExploreNodePair) {
  std::pair<int, int> nodePair;
  std::pair<int, int> arcPair;

  // For non-epsilon arc
  bool checkArcPair = false;
  // For epsilon arc
  std::pair<bool, bool> checkEpsilonArcPair = std::make_pair(false, false);

  bool isValid = false;

  // There should be at least two values to form a range
  assert(arcCrossProductOffset.size() >= 2);

  for (size_t i = 0; i < arcCrossProductOffset.size() - 1; ++i) {
    const int lVal = arcCrossProductOffset[i];
    const int rVal = arcCrossProductOffset[i + 1];

    if ((lVal <= tid) && (tid < rVal)) {
      isValid = true;
      nodePair = toExploreNodePair[i];

      // The range of idx is from
      // [0, toExploreNumArcs[i].first * toExploreNumArcs[i].second)
      const int idx = tid - lVal;
      const int numArcs = rVal - lVal;
      // arcCrossProductOffset[i + 1] - arcCrossProductOffset[i];

      assert(idx >= 0);
      assert(idx < numArcs);
      assert(numArcs > 0);

      const int arcProd =
          toExploreNumArcs[i].first * toExploreNumArcs[i].second;

      if (numArcs == arcProd) {
        checkArcPair = true;

        // We map the tids to 2D grid where the
        // x-axis is toExploreNumArcs[i].first (row)
        // y-axis is toExploreNumArcs[i].second (column)
        arcPair = OneDToTwoDIndex(idx, toExploreNumArcs[i].first);

        // Pick the tids from the first row since we need only one
        // tid per arc of the node from the first graph to check for
        // epsilon
        if (idx < toExploreNumArcs[i].first) {
          checkEpsilonArcPair.first = true;
        }

        // Pick the tids from the first column since we need only one
        // tid per arc of the node from the first graph to check for
        // epsilon
        if (idx % toExploreNumArcs[i].first == 0) {
          checkEpsilonArcPair.second = true;
        }
      } else if ((arcProd == 0) && (numArcs == toExploreNumArcs[i].first)) {
        // TODO: Likely not the brightest idea to use -1 as sentinel
        arcPair = std::make_pair(idx, -1);
        checkEpsilonArcPair.first = true;
      } else if ((arcProd == 0) && (numArcs == toExploreNumArcs[i].second)) {
        // TODO: Likely not the brightest idea to use -1 as sentinel
        arcPair = std::make_pair(-1, idx);
        checkEpsilonArcPair.second = true;
      }

      break;
    }
  }

  return std::make_tuple(
      isValid, nodePair, arcPair, checkArcPair, checkEpsilonArcPair);
}

// Takes a pair of nodes, where each member of pair comes from a different
// graph and calculate a vector of number of arcs in the cross product of
// arcs outgoing from each pair.
// This should be a kernel call
std::tuple<std::vector<int>, std::vector<std::pair<int, int>>>
calculateArcCrossProductOffset(
    const std::vector<std::pair<int, int>>& toExploreNodePair,
    const GraphDataParallel& graphDP1,
    const GraphDataParallel& graphDP2,
    bool inOrOutArc) {
  std::vector<std::pair<int, int>> toExploreNumArcs(toExploreNodePair.size());
  std::vector<int> arcCrossProductOffset(toExploreNodePair.size());

  // No dependence between iterations
  for (size_t i = 0; i < toExploreNodePair.size(); ++i) {
    int node = toExploreNodePair[i].first;
    // Special case if it is the last node. Then the offset becomes
    // the number of arcs
    const int inArcOffsetGraph1 = ((node + 1) == graphDP1.inArcOffset.size())
        ? graphDP1.inArcs.size()
        : graphDP1.inArcOffset[node + 1];
    const int outArcOffsetGraph1 = ((node + 1) == graphDP1.outArcOffset.size())
        ? graphDP1.outArcs.size()
        : graphDP1.outArcOffset[node + 1];

    const int numArcsFirst = inOrOutArc
        ? inArcOffsetGraph1 - graphDP1.inArcOffset[node]
        : outArcOffsetGraph1 - graphDP1.outArcOffset[node];

    node = toExploreNodePair[i].second;
    // Special case if it is the last node. Then the offset becomes
    // the number of arcs
    const int inArcOffsetGraph2 = ((node + 1) == graphDP2.inArcOffset.size())
        ? graphDP2.inArcs.size()
        : graphDP2.inArcOffset[node + 1];
    const int outArcOffsetGraph2 = ((node + 1) == graphDP2.outArcOffset.size())
        ? graphDP2.outArcs.size()
        : graphDP2.outArcOffset[node + 1];

    const int numArcsSecond = inOrOutArc
        ? inArcOffsetGraph2 - graphDP2.inArcOffset[node]
        : outArcOffsetGraph2 - graphDP2.outArcOffset[node];

    toExploreNumArcs[i] = std::make_pair(numArcsFirst, numArcsSecond);

    // Even when numArcsFirst or numArcsSecond is 0 we have to consider
    // the case when the other graph has arcs with epsilon label
    if (numArcsFirst != 0 && numArcsSecond != 0) {
      arcCrossProductOffset[i] = numArcsFirst * numArcsSecond;
    } else if (numArcsFirst != 0 && numArcsSecond == 0) {
      /*
      const int startIdx =
          inOrOutArc ? graphDP1.inArcOffset[node] : graphDP1.outArcOffset[node];
      const int endIdx = inOrOutArc ? inArcOffsetGraph1 : outArcOffsetGraph1;

      bool hasEpsilonArc = false;
      // BEWARE: Nested for loop!!!
      // Ideally this for will not exist and this information can be carried
      // as a per node flag - TBD
      // This loop loops over all arcs and checks if any of them has a epsilon
      // label
      for (int idx = startIdx; idx < endIdx; ++idx) {
        const int arcIdx =
            inOrOutArc ? graphDP1.inArcs[idx] : graphDP1.outArcs[idx];

        if (graphDP1.olabels[arcIdx] == epsilon) {
          hasEpsilonArc = true;
          break;
        }
      }

      arcCrossProductOffset[i] = hasEpsilonArc ? numArcsFirst : 0;*/
      arcCrossProductOffset[i] = numArcsFirst;
    } else if (numArcsFirst == 0 && numArcsSecond != 0) {
      /*
      const int startIdx =
          inOrOutArc ? graphDP2.inArcOffset[node] : graphDP2.outArcOffset[node];
      const int endIdx = inOrOutArc ? inArcOffsetGraph2 : outArcOffsetGraph2;

      bool hasEpsilonArc = false;
      // BEWARE: Nested for loop!!!
      // Ideally this for will not exist and this information can be carried
      // as a per node flag - TBD
      // This loop loops over all arcs and checks if any of them has a epsilon
      // label
      for (int idx = startIdx; idx < endIdx; ++idx) {
        const int arcIdx =
            inOrOutArc ? graphDP2.inArcs[idx] : graphDP2.outArcs[idx];

        if (graphDP2.ilabels[arcIdx] == epsilon) {
          hasEpsilonArc = true;
          break;
        }
      }

      arcCrossProductOffset[i] = hasEpsilonArc ? numArcsSecond : 0;*/
      arcCrossProductOffset[i] = numArcsSecond;
    } else {
      arcCrossProductOffset[i] = 0;
    }
  }

  return std::make_tuple(arcCrossProductOffset, toExploreNumArcs);
}

// This function needs to be thread safe since multiple threads can
// can call it and they will overlap on curIdx and dstIdx
void calculateNumArcsAndNodesToExplore(
    int curIdx,
    int dstIdx,
    const std::vector<bool>& reachable,
    std::vector<bool>& newNodes,
    std::vector<bool>& toExplore,
    std::vector<int>& numOutArcs,
    std::vector<int>& numInArcs) {
  if (reachable[dstIdx]) {
    // Atomic test and set for newNodes
    if (!newNodes[dstIdx]) {
      newNodes[dstIdx] = true;
      toExplore[dstIdx] = true;
    }

    // These are atomic increments
    numOutArcs[curIdx]++;
    numInArcs[dstIdx]++;
  }
}

// This function needs to be thread safe since multiple threads can
// can call it
void generateCombinedGraphNodesAndArcs(
    int dstIdx,
    int curIdx,
    const std::pair<int, int>& arcPair,
    const std::vector<bool>& reachable,
    const std::vector<int>& newNodesOffset,
    std::vector<bool>& newNodesVisited,
    std::vector<bool>& toExplore,
    std::pair<std::vector<int>, std::vector<int>>& gradInfo,
    GraphDataParallel& newGraphDP,
    std::pair<bool, bool> srcNodeStartAndAccept,
    std::pair<bool, bool> dstNodeStartAndAccept,
    int ilabel,
    int olabel,
    float weight) {
  if (reachable[dstIdx]) {
    // Atomic test and set for newNodesVisited
    if (!newNodesVisited[dstIdx]) {
      newNodesVisited[dstIdx] = true;
      toExplore[dstIdx] = true;
    }

    // Set accept and start nodes
    // I think I only need it for dst nodes and src nodes
    // Note: Multiple threads can have the same dstIdx and write to the same
    //       location and collide. This _should_ be fine since they are going
    //       to write the same value
    newGraphDP.start[newNodesOffset[dstIdx]] = dstNodeStartAndAccept.first;
    newGraphDP.accept[newNodesOffset[dstIdx]] = dstNodeStartAndAccept.second;

    // Both of these increments are atomic
    int inArcIdx = newGraphDP.inArcOffset[newNodesOffset[dstIdx]]++;
    int outArcIdx = newGraphDP.outArcOffset[newNodesOffset[curIdx]]++;

    // outArcIdx is also the arc identifier
    newGraphDP.outArcs[outArcIdx] = outArcIdx;
    newGraphDP.inArcs[inArcIdx] = outArcIdx;

    // Fill in everything else for this arc
    newGraphDP.ilabels[outArcIdx] = ilabel;
    newGraphDP.olabels[outArcIdx] = olabel;
    newGraphDP.srcNodes[outArcIdx] = newNodesOffset[curIdx];
    newGraphDP.dstNodes[outArcIdx] = newNodesOffset[dstIdx];
    newGraphDP.weights[outArcIdx] = weight;

    gradInfo.first[outArcIdx] = arcPair.first;
    gradInfo.second[outArcIdx] = arcPair.second;
  }
}

// Convert bool array two pairs for true flags
std::vector<std::pair<int, int>> convertToNodePair(
    const std::vector<bool>& flags,
    int extent) {
  std::vector<std::pair<int, int>> toExploreNodePair;
  for (size_t i = 0; i < flags.size(); ++i) {
    if (flags[i] == true) {
      toExploreNodePair.push_back(OneDToTwoDIndex(i, extent));
    }
  }

  return toExploreNodePair;
}

// Takes a bool array with flags set for nodes to pick and returns
// an array with indices that were set as true
std::vector<int> convertToNodes(const std::vector<bool>& flags) {
  std::vector<int> nodes;

  for (size_t i = 0; i < flags.size(); ++i) {
    if (flags[i]) {
      nodes.push_back(i);
    }
  }

  return nodes;
}

std::tuple<std::pair<bool, bool>, std::pair<bool, bool>> getStartAndAccept(
    const GraphDataParallel& graphDP1,
    const GraphDataParallel& graphDP2,
    const std::pair<int, int>& srcNodePair,
    const std::pair<int, int>& dstNodePair) {
  const std::pair<bool, bool> srcNodeStartAndAccept = std::make_pair(
      graphDP1.start[srcNodePair.first] && graphDP2.start[srcNodePair.second],
      graphDP1.accept[srcNodePair.first] &&
          graphDP2.accept[srcNodePair.second]);

  const std::pair<bool, bool> dstNodeStartAndAccept = std::make_pair(
      graphDP1.start[dstNodePair.first] && graphDP2.start[dstNodePair.second],
      graphDP1.accept[dstNodePair.first] &&
          graphDP2.accept[dstNodePair.second]);

  return std::make_tuple(srcNodeStartAndAccept, dstNodeStartAndAccept);
}

} // namespace

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

Graph compose(const Graph& first, const Graph& second) {
  GraphDataParallel graphDP1, graphDP2;
  // Convert from AOS to SOA
  graphDP1 = convertToDataParallel(first);
  graphDP2 = convertToDataParallel(second);

  //////////////////////////////////////////////////////////////////////////
  // Step 1: Data parallel findReachable
  //////////////////////////////////////////////////////////////////////////
  std::vector<bool> reachable(first.numNodes() * second.numNodes(), false);

  std::vector<bool> toExplore(first.numNodes() * second.numNodes(), false);
  // std::vector<bool> epsilonMatched(first.numNodes() * second.numNodes(), false);

  const int numNodesFirst = first.numNodes();

  {
    std::vector<int> acceptDP1 = convertToNodes(graphDP1.accept);
    std::vector<int> acceptDP2 = convertToNodes(graphDP2.accept);

    for (auto f : acceptDP1) {
      for (auto s : acceptDP2) {
        toExplore[TwoDToOneDIndex(f, s, numNodesFirst)] = true;
        reachable[TwoDToOneDIndex(f, s, numNodesFirst)] = true;
      }
    }
  }

  // This is the outer control loop that would spawn DP kernels
  while (checkAnyTrue(toExplore)) {
    // Convert bits set in toExplore to node pairs
    auto toExploreNodePair = convertToNodePair(toExplore, numNodesFirst);

    // Reset so pristine state for next frontier to explore
    // No dependence between iterations
    std::fill(toExplore.begin(), toExplore.end(), false);
    // std::fill(epsilonMatched.begin(), epsilonMatched.end(), false);

    std::vector<int> arcCrossProductOffset;
    std::vector<std::pair<int, int>> toExploreNumArcs;
    std::tie(arcCrossProductOffset, toExploreNumArcs) =
        calculateArcCrossProductOffset(
            toExploreNodePair, graphDP1, graphDP2, true);

    const int totalArcs = prefixSumScan(arcCrossProductOffset, true);

    // No dependence between iterations. tid is thread-id
    // Only do non epsilon case for this kernel
    for (int tid = 0; tid < totalArcs; ++tid) {
      // Map tid to corresponding node and arc pair via search
      std::pair<int, int> nodePair;
      std::pair<int, int> arcPair;
      bool checkArcPair;
      std::pair<bool, bool> checkEpsilonArcPair;
      bool isValid;
      std::tie(isValid, nodePair, arcPair, checkArcPair, checkEpsilonArcPair) =
          computeNodeAndArcPair(
              tid, arcCrossProductOffset, toExploreNumArcs, toExploreNodePair);

      // Does this node pair match?
      if (isValid) {
        int inArcOffset = graphDP1.inArcOffset[nodePair.first];
        const int firstArcIdx = graphDP1.inArcs[inArcOffset + arcPair.first];

        inArcOffset = graphDP2.inArcOffset[nodePair.second];
        const int secondArcIdx = graphDP2.inArcs[inArcOffset + arcPair.second];

        if (checkArcPair &&
            (graphDP1.olabels[firstArcIdx] == graphDP2.ilabels[secondArcIdx])) {
          const int idx = TwoDToOneDIndex(
              graphDP1.srcNodes[firstArcIdx],
              graphDP2.srcNodes[secondArcIdx],
              numNodesFirst);
          // idx may not be unique amongst all threads. In particular
          // if two pairs of arcs that have same olabel and ilabel then idx
          // won't be unique and this is a race but both would mark the
          // destination node as reachable
          if (!reachable[idx]) {
            toExplore[idx] = true;
          }
          reachable[idx] = true;

          // We track if any two arcs incoming to this pair of nodes matched
          // on epsilon
          /*
          if (graphDP1.olabels[firstArcIdx] == epsilon) {
            epsilonMatched[TwoDToOneDIndex(
                nodePair.first, nodePair.second, numNodesFirst)] = true;
          }*/
        }
      }
    }

    // No dependence between iterations. tid is thread-id
    // Do epsilon match case in this kernel launch
    for (int tid = 0; tid < totalArcs; ++tid) {
      // Map tid to corresponding node and arc pair via search
      std::pair<int, int> nodePair;
      std::pair<int, int> arcPair;
      bool checkArcPair;
      std::pair<bool, bool> checkEpsilonArcPair;
      bool isValid;
      std::tie(isValid, nodePair, arcPair, checkArcPair, checkEpsilonArcPair) =
          computeNodeAndArcPair(
              tid, arcCrossProductOffset, toExploreNumArcs, toExploreNodePair);
      // const bool matched = epsilonMatched[TwoDToOneDIndex(
          // nodePair.first, nodePair.second, numNodesFirst)];

      // if (isValid && !matched) {
      if (isValid) {
        int inArcOffset = graphDP1.inArcOffset[nodePair.first];
        const int firstArcIdx = graphDP1.inArcs[inArcOffset + arcPair.first];

        inArcOffset = graphDP2.inArcOffset[nodePair.second];
        const int secondArcIdx = graphDP2.inArcs[inArcOffset + arcPair.second];

        // Only valid for arcs incoming to node from first graph
        if (checkEpsilonArcPair.first &&
            (graphDP1.olabels[firstArcIdx] == epsilon)) {
          const int idx = TwoDToOneDIndex(
              graphDP1.srcNodes[firstArcIdx], nodePair.second, numNodesFirst);
          if (!reachable[idx]) {
            toExplore[idx] = true;
          }
          reachable[idx] = true;
        }

        // Only valid for arcs incoming to node from second graph
        if (checkEpsilonArcPair.second &&
            (graphDP2.ilabels[secondArcIdx] == epsilon)) {
          const int idx = TwoDToOneDIndex(
              nodePair.first, graphDP2.srcNodes[secondArcIdx], numNodesFirst);
          if (!reachable[idx]) {
            toExplore[idx] = true;
          }
          reachable[idx] = true;
        }
      }
    }
  } // end while for findReachable

  //////////////////////////////////////////////////////////////////////////
  // Step 3: Compute a) valid nodes in combined graph
  //                 b) Number of in and out arcs in combined graph
  // This information is used to generate offsets for nodes and arcs
  // in the combined graph
  //////////////////////////////////////////////////////////////////////////
  std::vector<bool> newNodes(first.numNodes() * second.numNodes(), false);
  std::vector<bool> epsilonMatchedF(first.numNodes() * second.numNodes(), false);

  // Number of in and out arcs per node
  std::vector<int> numOutArcs(first.numNodes() * second.numNodes(), 0);
  std::vector<int> numInArcs(first.numNodes() * second.numNodes(), 0);

  // Tracks the nodes that are going to be present in the combined graph
  // std::fill(newNodes.begin(), newNodes.end(), false);
  std::fill(toExplore.begin(), toExplore.end(), false);

  {
    std::vector<int> startDP1 = convertToNodes(graphDP1.start);
    std::vector<int> startDP2 = convertToNodes(graphDP2.start);

    for (auto f : startDP1) {
      for (auto s : startDP2) {
        auto startIdx = TwoDToOneDIndex(f, s, numNodesFirst);
        if (reachable[startIdx]) {
          toExplore[startIdx] = true;
          newNodes[startIdx] = true;
        }
      }
    }
  }

  // This is the outer control loop that would spawn DP kernels
  while (checkAnyTrue(toExplore)) {
    // Convert bits set in toExplore to node pairs
    auto toExploreNodePair = convertToNodePair(toExplore, numNodesFirst);

    // Reset to pristine state for next frontier to explore
    std::fill(toExplore.begin(), toExplore.end(), false);
    std::fill(epsilonMatchedF.begin(), epsilonMatchedF.end(), false);

    std::vector<int> arcCrossProductOffset;
    std::vector<std::pair<int, int>> toExploreNumArcs;
    std::tie(arcCrossProductOffset, toExploreNumArcs) =
        calculateArcCrossProductOffset(
            toExploreNodePair, graphDP1, graphDP2, false);

    const int totalArcs = prefixSumScan(arcCrossProductOffset, true);

    // No dependence between iterations
    for (int tid = 0; tid < totalArcs; ++tid) {
      // Map tid to corresponding node and arc pair
      // Search to find which node pair this tid will fall into
      std::pair<int, int> nodePair;
      std::pair<int, int> arcPair;
      bool checkArcPair;
      std::pair<bool, bool> checkEpsilonArcPair;
      bool isValid;
      std::tie(isValid, nodePair, arcPair, checkArcPair, checkEpsilonArcPair) =
          computeNodeAndArcPair(
              tid, arcCrossProductOffset, toExploreNumArcs, toExploreNodePair);

      if (isValid) {
        int outArcOffset = graphDP1.outArcOffset[nodePair.first];
        const int firstArcIdx = graphDP1.outArcs[outArcOffset + arcPair.first];

        outArcOffset = graphDP2.outArcOffset[nodePair.second];
        const int secondArcIdx =
            graphDP2.outArcs[outArcOffset + arcPair.second];

        // Does this node pair match?
        // Skip epsilon matches
        if (checkArcPair &&
            (graphDP1.olabels[firstArcIdx] == graphDP2.ilabels[secondArcIdx])) {
          const int dstIdx = TwoDToOneDIndex(
              graphDP1.dstNodes[firstArcIdx],
              graphDP2.dstNodes[secondArcIdx],
              numNodesFirst);
          const int curIdx =
              TwoDToOneDIndex(nodePair.first, nodePair.second, numNodesFirst);

          // We track if any two arcs outgoing from this node pair match
          // on epsilon. We record if they do.
          if (graphDP1.olabels[firstArcIdx] == epsilon) {
            epsilonMatchedF[TwoDToOneDIndex(
                nodePair.first, nodePair.second, numNodesFirst)] = true;
          } else {
            calculateNumArcsAndNodesToExplore(
                curIdx,
                dstIdx,
                reachable,
                newNodes,
                toExplore,
                numOutArcs,
                numInArcs);
          }
        }
      }
    }

    for (int tid = 0; tid < totalArcs; ++tid) {
      // Map tid to corresponding node and arc pair
      // Search to find which node pair this tid will fall into
      std::pair<int, int> nodePair;
      std::pair<int, int> arcPair;
      bool checkArcPair;
      std::pair<bool, bool> checkEpsilonArcPair;
      bool isValid;
      std::tie(isValid, nodePair, arcPair, checkArcPair, checkEpsilonArcPair) =
          computeNodeAndArcPair(
              tid, arcCrossProductOffset, toExploreNumArcs, toExploreNodePair);

      if (isValid) {
        int outArcOffset = graphDP1.outArcOffset[nodePair.first];
        const int firstArcIdx = graphDP1.outArcs[outArcOffset + arcPair.first];

        outArcOffset = graphDP2.outArcOffset[nodePair.second];
        const int secondArcIdx =
            graphDP2.outArcs[outArcOffset + arcPair.second];

        const bool epsilonMatch = epsilonMatchedF[TwoDToOneDIndex(
            nodePair.first, nodePair.second, numNodesFirst)];

        if (checkEpsilonArcPair.first &&
            (!epsilonMatch || graphDP2.accept[nodePair.second] ||
            !graphDP1.accept[nodePair.first]) &&
            (graphDP1.olabels[firstArcIdx] == epsilon)) {
          const int dstIdx = TwoDToOneDIndex(
              graphDP1.dstNodes[firstArcIdx], nodePair.second, numNodesFirst);
          const int curIdx =
              TwoDToOneDIndex(nodePair.first, nodePair.second, numNodesFirst);

          calculateNumArcsAndNodesToExplore(
              curIdx,
              dstIdx,
              reachable,
              newNodes,
              toExplore,
              numOutArcs,
              numInArcs);
        }

        if (checkEpsilonArcPair.second &&
            (!epsilonMatch || graphDP1.accept[nodePair.first]) &&
            (graphDP2.ilabels[secondArcIdx] == epsilon)) {
          const int dstIdx = TwoDToOneDIndex(
              nodePair.first, graphDP2.dstNodes[secondArcIdx], numNodesFirst);
          const int curIdx =
              TwoDToOneDIndex(nodePair.first, nodePair.second, numNodesFirst);

          calculateNumArcsAndNodesToExplore(
              curIdx,
              dstIdx,
              reachable,
              newNodes,
              toExplore,
              numOutArcs,
              numInArcs);
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////
  // Step 4: Generate offsets for nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////
  // Generate offsets for nodes and arcs
  GraphDataParallel newGraphDP;

  // Convert bool array to int for prefix sum
  // Record arc offsets for new nodes in new graph
  std::vector<int> newNodesOffset(newNodes.size(), 0);
  for (size_t i = 0; i < newNodes.size(); ++i) {
    if (newNodes[i]) {
      newNodesOffset[i] = 1;
      newGraphDP.inArcOffset.push_back(numInArcs[i]);
      newGraphDP.outArcOffset.push_back(numOutArcs[i]);
    }
  }

  const int totalNodes = prefixSumScan(newNodesOffset, false);

  // Check that number of nodes match
  assert(totalNodes == newGraphDP.inArcOffset.size());
  assert(newGraphDP.inArcOffset.size() == newGraphDP.outArcOffset.size());

  // Prefix sum to generate offsets
  const int totalInArcs = prefixSumScan(newGraphDP.inArcOffset, false);
  const int totalOutArcs = prefixSumScan(newGraphDP.outArcOffset, false);

  // Allocate space for start and accept nodes
  assert(newGraphDP.start.empty());
  assert(newGraphDP.accept.empty());
  newGraphDP.start.resize(totalNodes, false);
  newGraphDP.accept.resize(totalNodes, false);

  // This is the total number of arcs and they must be equal
  assert(totalInArcs == totalOutArcs);

  newGraphDP.inArcs.resize(totalInArcs);
  newGraphDP.outArcs.resize(totalOutArcs);
  newGraphDP.ilabels.resize(totalOutArcs);
  newGraphDP.olabels.resize(totalOutArcs);
  newGraphDP.srcNodes.resize(totalOutArcs);
  newGraphDP.dstNodes.resize(totalOutArcs);
  newGraphDP.weights.resize(totalOutArcs);

  // SOA for gradInfo
  std::pair<std::vector<int>, std::vector<int>> gradInfo;
  gradInfo.first.resize(totalOutArcs);
  gradInfo.second.resize(totalOutArcs);

  //////////////////////////////////////////////////////////////////////////
  // Step 5: Generate nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////
  std::fill(toExplore.begin(), toExplore.end(), false);
  std::fill(epsilonMatchedF.begin(), epsilonMatchedF.end(), false);
  std::vector<bool> newNodesVisited(
      first.numNodes() * second.numNodes(), false);

  {
    std::vector<int> startDP1 = convertToNodes(graphDP1.start);
    std::vector<int> startDP2 = convertToNodes(graphDP2.start);

    for (auto f : startDP1) {
      for (auto s : startDP2) {
        const int nodeIdx = TwoDToOneDIndex(f, s, numNodesFirst);
        if (reachable[nodeIdx]) {
          toExplore[nodeIdx] = true;
          newNodesVisited[nodeIdx] = true;
          newGraphDP.start[newNodesOffset[nodeIdx]] = true;
          newGraphDP.accept[newNodesOffset[nodeIdx]] =
              graphDP1.accept[f] && graphDP1.accept[s];
        }
      }
    }
  }

  // This is the outer control loop that would spawn DP kernels
  while (checkAnyTrue(toExplore)) {
    // Convert bits set in toExplore to node pairs
    auto toExploreNodePair = convertToNodePair(toExplore, numNodesFirst);

    // Reset so pristine state for next frontier to explore
    // No dependence between iterations
    std::fill(toExplore.begin(), toExplore.end(), false);
    std::fill(epsilonMatchedF.begin(), epsilonMatchedF.end(), false);

    std::vector<int> arcCrossProductOffset;
    std::vector<std::pair<int, int>> toExploreNumArcs;
    std::tie(arcCrossProductOffset, toExploreNumArcs) =
        calculateArcCrossProductOffset(
            toExploreNodePair, graphDP1, graphDP2, false);

    const int totalArcs = prefixSumScan(arcCrossProductOffset, true);

    // No dependence between iterations
    for (int tid = 0; tid < totalArcs; ++tid) {
      // Map tid to corresponding node and arc pair
      // Search to find which node pair this tid will fall into
      std::pair<int, int> srcNodePair;
      std::pair<int, int> arcPair;
      bool checkArcPair;
      std::pair<bool, bool> checkEpsilonArcPair;
      bool isValid;
      std::tie(
          isValid, srcNodePair, arcPair, checkArcPair, checkEpsilonArcPair) =
          computeNodeAndArcPair(
              tid, arcCrossProductOffset, toExploreNumArcs, toExploreNodePair);

      if (isValid) {
        int outArcOffset = graphDP1.outArcOffset[srcNodePair.first];
        const int firstArcIdx = graphDP1.outArcs[outArcOffset + arcPair.first];

        outArcOffset = graphDP2.outArcOffset[srcNodePair.second];
        const int secondArcIdx =
            graphDP2.outArcs[outArcOffset + arcPair.second];

        // Does this node pair match?
        if (checkArcPair &&
            (graphDP1.olabels[firstArcIdx] == graphDP2.ilabels[secondArcIdx])) {
          std::pair<int, int> dstNodePair = std::make_pair(
              graphDP1.dstNodes[firstArcIdx], graphDP2.dstNodes[secondArcIdx]);

          const int dstIdx = TwoDToOneDIndex(
              dstNodePair.first, dstNodePair.second, numNodesFirst);
          const int curIdx = TwoDToOneDIndex(
              srcNodePair.first, srcNodePair.second, numNodesFirst);

          std::pair<bool, bool> srcNodeStartAndAccept;
          std::pair<bool, bool> dstNodeStartAndAccept;

          std::tie(srcNodeStartAndAccept, dstNodeStartAndAccept) =
              getStartAndAccept(graphDP1, graphDP2, srcNodePair, dstNodePair);

          // We track if any two arcs outgoing from this node pair match
          // on epsilon. We record if they do.
          if (graphDP1.olabels[firstArcIdx] == epsilon) {
            epsilonMatchedF[TwoDToOneDIndex(
                srcNodePair.first, srcNodePair.second, numNodesFirst)] = true;
          } else {
            generateCombinedGraphNodesAndArcs(
                dstIdx,
                curIdx,
                std::make_pair(firstArcIdx, secondArcIdx),
                reachable,
                newNodesOffset,
                newNodesVisited,
                toExplore,
                gradInfo,
                newGraphDP,
                srcNodeStartAndAccept,
                dstNodeStartAndAccept,
                graphDP1.ilabels[firstArcIdx],
                graphDP2.olabels[secondArcIdx],
                graphDP1.weights[firstArcIdx] + graphDP2.weights[secondArcIdx]);
          }
        }
      }
    }

    // No dependence between iterations
    for (int tid = 0; tid < totalArcs; ++tid) {
      // Map tid to corresponding node and arc pair
      // Search to find which node pair this tid will fall into
      std::pair<int, int> srcNodePair;
      std::pair<int, int> arcPair;
      bool checkArcPair;
      std::pair<bool, bool> checkEpsilonArcPair;
      bool isValid;
      std::tie(
          isValid, srcNodePair, arcPair, checkArcPair, checkEpsilonArcPair) =
          computeNodeAndArcPair(
              tid, arcCrossProductOffset, toExploreNumArcs, toExploreNodePair);

      if (isValid) {
        int outArcOffset = graphDP1.outArcOffset[srcNodePair.first];
        const int firstArcIdx = graphDP1.outArcs[outArcOffset + arcPair.first];

        outArcOffset = graphDP2.outArcOffset[srcNodePair.second];
        const int secondArcIdx =
            graphDP2.outArcs[outArcOffset + arcPair.second];

        const bool epsilonMatch = epsilonMatchedF[TwoDToOneDIndex(
            srcNodePair.first, srcNodePair.second, numNodesFirst)];

        // The epsilon matches
        if (checkEpsilonArcPair.first &&
            (!epsilonMatch || graphDP2.accept[srcNodePair.second] ||
            !graphDP1.accept[srcNodePair.first]) &&
            (graphDP1.olabels[firstArcIdx] == epsilon)) {
          // When arc from first node has epsilon label then we consider
          // second node
          std::pair<int, int> dstNodePair = std::make_pair(
              graphDP1.dstNodes[firstArcIdx], srcNodePair.second);
          const int dstIdx = TwoDToOneDIndex(
              dstNodePair.first, dstNodePair.second, numNodesFirst);
          const int curIdx = TwoDToOneDIndex(
              srcNodePair.first, srcNodePair.second, numNodesFirst);

          std::pair<bool, bool> srcNodeStartAndAccept;
          std::pair<bool, bool> dstNodeStartAndAccept;

          std::tie(srcNodeStartAndAccept, dstNodeStartAndAccept) =
              getStartAndAccept(graphDP1, graphDP2, srcNodePair, dstNodePair);
          generateCombinedGraphNodesAndArcs(
              dstIdx,
              curIdx,
              std::make_pair(firstArcIdx, -1),
              reachable,
              newNodesOffset,
              newNodesVisited,
              toExplore,
              gradInfo,
              newGraphDP,
              srcNodeStartAndAccept,
              dstNodeStartAndAccept,
              graphDP1.ilabels[firstArcIdx],
              epsilon,
              graphDP1.weights[firstArcIdx]);
        }

        // The epsilon matches
        if (checkEpsilonArcPair.second &&
            (!epsilonMatch || graphDP1.accept[srcNodePair.first]) &&
            (graphDP2.ilabels[secondArcIdx] == epsilon)) {
          // When arc from second node has epsilon label then we consider
          // first node
          std::pair<int, int> dstNodePair = std::make_pair(
              srcNodePair.first, graphDP2.dstNodes[secondArcIdx]);
          const int dstIdx = TwoDToOneDIndex(
              dstNodePair.first, dstNodePair.second, numNodesFirst);
          const int curIdx = TwoDToOneDIndex(
              srcNodePair.first, srcNodePair.second, numNodesFirst);

          std::pair<bool, bool> srcNodeStartAndAccept;
          std::pair<bool, bool> dstNodeStartAndAccept;

          std::tie(srcNodeStartAndAccept, dstNodeStartAndAccept) =
              getStartAndAccept(graphDP1, graphDP2, srcNodePair, dstNodePair);

          generateCombinedGraphNodesAndArcs(
              dstIdx,
              curIdx,
              std::make_pair(-1, secondArcIdx),
              reachable,
              newNodesOffset,
              newNodesVisited,
              toExplore,
              gradInfo,
              newGraphDP,
              srcNodeStartAndAccept,
              dstNodeStartAndAccept,
              epsilon,
              graphDP2.olabels[secondArcIdx],
              graphDP2.weights[secondArcIdx]);
        }
      }
    }
  }
  // Shift offset values back down after adding arcs to newGraphDP
  // The offset values got converted from exclusive prefix sum to inclusive
  // Need to convert them back to exclusive prefix sum  by starting with 0
  // and shifting to right by 1
  for (int i = newGraphDP.outArcOffset.size() - 1; i >= 0; --i) {
    newGraphDP.outArcOffset[i] = i == 0 ? 0 : newGraphDP.outArcOffset[i - 1];
    newGraphDP.inArcOffset[i] = i == 0 ? 0 : newGraphDP.inArcOffset[i - 1];
  }

  // Convert back and add in autograd metadata
  auto nGraph = convertFromDataParallel(newGraphDP);
  nGraph.setInputs({first, second});
  // Convert gradInfo SOA to AOS
  std::vector<std::pair<int, int>> gradInfoAOS;
  for (int i = 0; i < gradInfo.first.size(); ++i) {
    gradInfoAOS.emplace_back(gradInfo.first[i], gradInfo.second[i]);
  }
  // TODO eliminate this copy pasta.
  auto gradFunc = [gradInfo = std::move(gradInfoAOS)](
                      std::vector<Graph>& inputs, Graph deltas) {
    // In this case the arc's parents are always from the
    // first and second input graphs respectively.
    bool calcGrad1 = inputs[0].calcGrad();
    bool calcGrad2 = inputs[1].calcGrad();
    auto grad1 = calcGrad1 ? std::vector<float>(inputs[0].numArcs(), 0.0)
                           : std::vector<float>{};
    auto grad2 = calcGrad2 ? std::vector<float>(inputs[1].numArcs(), 0.0)
                           : std::vector<float>{};
    for (int i = 0; i < gradInfo.size(); i++) {
      auto arcGrad = deltas.weight(i);
      auto& arcs = gradInfo[i];
      if (calcGrad1 && arcs.first >= 0) {
        grad1[arcs.first] += arcGrad;
      }
      if (calcGrad2 && arcs.second >= 0) {
        grad2[arcs.second] += arcGrad;
      }
    }
    inputs[0].addGrad(std::move(grad1));
    inputs[1].addGrad(std::move(grad2));
  };
  nGraph.setGradFunc(std::move(gradFunc));
  return nGraph;
}

} // namespace dataparallel
} // namespace detail
} // namespace gtn
