/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
#include <tuple>

#include "parallel_compose.h"
#include "prefix_scan.h"

namespace gtn {
namespace detail {
namespace dataparallel {

namespace {

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

bool checkAnyTrue(const std::vector<int>& flags) {
  // Potentially wasteful - but GPU friendly
  return std::accumulate(flags.begin(), flags.end(), 0) > 0 ? true : false;
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
    const std::pair<std::vector<int>, std::vector<int>>& toExploreNumArcs,
    const std::pair<std::vector<int>, std::vector<int>>& toExploreNodePair) {
  assert(toExploreNodePair.first.size() == toExploreNodePair.second.size());
  assert(arcCrossProductOffset.size() == (toExploreNodePair.first.size() + 1));
  assert(toExploreNumArcs.first.size() == toExploreNumArcs.second.size());

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
      nodePair = std::make_pair(
          (toExploreNodePair.first)[i], (toExploreNodePair.second)[i]);

      // The range of idx is from
      // [0, toExploreNumArcs.first[i] * toExploreNumArcs.second[i])
      const int idx = tid - lVal;
      const int numArcs = rVal - lVal;

      assert(idx >= 0);
      assert(idx < numArcs);
      assert(numArcs > 0);

      const int arcProd =
          (toExploreNumArcs.first)[i] * (toExploreNumArcs.second)[i];

      if (numArcs == arcProd) {
        checkArcPair = true;

        // We map the tids to 2D grid where the
        // x-axis is toExploreNumArcs.first[i] (row)
        // y-axis is toExploreNumArcs.second[i] (column)
        arcPair = OneDToTwoDIndex(idx, (toExploreNumArcs.first)[i]);

        // Pick the tids from the first row since we need only one
        // tid per arc of the node from the first graph to check for
        // epsilon
        if (idx < (toExploreNumArcs.first)[i]) {
          checkEpsilonArcPair.first = true;
        }

        // Pick the tids from the first column since we need only one
        // tid per arc of the node from the first graph to check for
        // epsilon
        if (idx % (toExploreNumArcs.first)[i] == 0) {
          checkEpsilonArcPair.second = true;
        }
      } else if ((arcProd == 0) && (numArcs == (toExploreNumArcs.first)[i])) {
        // TODO: Likely not the brightest idea to use -1 as sentinel
        arcPair = std::make_pair(idx, -1);
        checkEpsilonArcPair.first = true;
      } else if ((arcProd == 0) && (numArcs == (toExploreNumArcs.second)[i])) {
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
std::tuple<std::vector<int>, std::pair<std::vector<int>, std::vector<int>>>
calculateArcCrossProductOffset(
    const std::pair<std::vector<int>, std::vector<int>>& toExploreNodePair,
    const GraphDataParallel& graphDP1,
    const GraphDataParallel& graphDP2,
    bool inOrOutArc) {
  assert(toExploreNodePair.first.size() == toExploreNodePair.second.size());

  std::pair<std::vector<int>, std::vector<int>> toExploreNumArcs;
  toExploreNumArcs.first.resize(toExploreNodePair.first.size());
  toExploreNumArcs.second.resize(toExploreNodePair.first.size());

  std::vector<int> arcCrossProductOffset(toExploreNodePair.first.size());

  // No dependence between iterations
  for (size_t i = 0; i < toExploreNodePair.first.size(); ++i) {
    int node = (toExploreNodePair.first)[i];
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

    node = (toExploreNodePair.second)[i];
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

    (toExploreNumArcs.first)[i] = numArcsFirst;
    (toExploreNumArcs.second)[i] = numArcsSecond;

    // Even when numArcsFirst or numArcsSecond is 0 we have to consider
    // the case when the other graph has arcs with epsilon label
    if (numArcsFirst != 0 && numArcsSecond != 0) {
      arcCrossProductOffset[i] = numArcsFirst * numArcsSecond;
    } else if (numArcsFirst != 0 && numArcsSecond == 0) {
      arcCrossProductOffset[i] = numArcsFirst;
    } else if (numArcsFirst == 0 && numArcsSecond != 0) {
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
    const std::vector<int>& reachable,
    std::vector<int>& newNodes,
    std::vector<int>& toExplore,
    std::vector<int>& numOutArcs,
    std::vector<int>& numInArcs) {
  if (reachable[dstIdx]) {
    // Atomic test and set (CAS) for newNodes
    int oldVal = newNodes[dstIdx];
    if (!newNodes[dstIdx]) {
      newNodes[dstIdx] = true;
    }

    if (!oldVal) {
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
    const std::pair<bool, bool>& dstNodeStartAndAccept,
    const std::vector<int>& reachable,
    const std::vector<int>& newNodesOffset,
    std::vector<int>& newNodesVisited,
    std::vector<int>& toExplore,
    std::pair<std::vector<int>, std::vector<int>>& gradInfo,
    GraphDataParallel& newGraphDP,
    int ilabel,
    int olabel,
    float weight) {
  if (reachable[dstIdx]) {
    // Atomic test and set (CAS) for newNodesVisited
    int oldVal = newNodesVisited[dstIdx];
    if (!newNodesVisited[dstIdx]) {
      newNodesVisited[dstIdx] = true;
    }

    if (!oldVal) {
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
std::pair<std::vector<int>, std::vector<int>> convertToNodePair(
    const std::vector<int>& flags,
    int extent) {
  std::vector<int> indices(flags);
  const int numValidNodes = prefixSumScan(indices, false);

  std::vector<int> toExploreNodePairFirst(numValidNodes);
  std::vector<int> toExploreNodePairSecond(numValidNodes);

  // No loop dependence
  for (size_t i = 0; i < flags.size(); ++i) {
    if (flags[i] == true) {
      std::pair<int, int> node = OneDToTwoDIndex(i, extent);

      const int index = indices[i];
      assert(index >= 0);
      assert(index < numValidNodes);
      toExploreNodePairFirst[index] = node.first;
      toExploreNodePairSecond[index] = node.second;
    }
  }

  return std::make_pair(toExploreNodePairFirst, toExploreNodePairSecond);
}

std::pair<bool, bool> getStartAndAccept(
    const GraphDataParallel& graphDP1,
    const GraphDataParallel& graphDP2,
    const std::pair<int, int>& dstNodePair) {
  const std::pair<bool, bool> dstNodeStartAndAccept = std::make_pair(
      graphDP1.start[dstNodePair.first] && graphDP2.start[dstNodePair.second],
      graphDP1.accept[dstNodePair.first] &&
          graphDP2.accept[dstNodePair.second]);

  return dstNodeStartAndAccept;
}

} // namespace

Graph compose(const Graph& first, const Graph& second) {
  GraphDataParallel graphDP1, graphDP2;
  // Convert from AOS to SOA
  graphDP1 = convertToDataParallel(first);
  graphDP2 = convertToDataParallel(second);

  const int numAllPairNodes = first.numNodes() * second.numNodes();
  const int numNodesFirst = first.numNodes();

  //////////////////////////////////////////////////////////////////////////
  // Step 1: Data parallel findReachable
  //////////////////////////////////////////////////////////////////////////
  std::vector<int> reachable(numAllPairNodes, false);
  std::vector<int> epsilonMatched(numAllPairNodes, false);

  std::vector<int> toExplore(numAllPairNodes, false);

  {
    for (int i = 0; i < numAllPairNodes; ++i) {
      std::pair<int, int> indices = OneDToTwoDIndex(i, numNodesFirst);

      if (graphDP1.accept[indices.first] && graphDP2.accept[indices.second]) {
        toExplore[i] = true;
        reachable[i] = true;
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

    std::vector<int> arcCrossProductOffset;
    std::pair<std::vector<int>, std::vector<int>> toExploreNumArcs;
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

          if (graphDP1.olabels[firstArcIdx] == epsilon) {
            epsilonMatched[idx] = true;
          }

          // idx may not be unique amongst all threads. In particular
          // if two pairs of arcs that have same olabel and ilabel then idx
          // won't be unique and this is a race but both would mark the
          // destination node as reachable
          int oldVal = reachable[idx];
          if (!reachable[idx]) {
            reachable[idx] = true;
          }
          if (!oldVal) {
            toExplore[idx] = true;
          }
        }

        // Only valid for arcs incoming to node from first graph
        if (checkEpsilonArcPair.first &&
            (graphDP1.olabels[firstArcIdx] == epsilon)) {
          const int idx = TwoDToOneDIndex(
              graphDP1.srcNodes[firstArcIdx], nodePair.second, numNodesFirst);
          int oldVal = reachable[idx];
          if (!reachable[idx]) {
            reachable[idx] = true;
          }
          if (!oldVal) {
            toExplore[idx] = true;
          }
        }

        // Only valid for arcs incoming to node from second graph
        if (checkEpsilonArcPair.second &&
            (graphDP2.ilabels[secondArcIdx] == epsilon)) {
          const int idx = TwoDToOneDIndex(
              nodePair.first, graphDP2.srcNodes[secondArcIdx], numNodesFirst);
          int oldVal = reachable[idx];
          if (!reachable[idx]) {
            reachable[idx] = true;
          }
          if (!oldVal) {
            toExplore[idx] = true;
          }
        }
      }
    }
  } // end while for findReachable

  //////////////////////////////////////////////////////////////////////////
  // Step 2: Compute a) valid nodes in combined graph
  //                 b) Number of in and out arcs in combined graph
  // This information is used to generate offsets for nodes and arcs
  // in the combined graph
  //////////////////////////////////////////////////////////////////////////
  std::vector<int> newNodes(numAllPairNodes, false);

  // Number of in and out arcs per node
  std::vector<int> numOutArcs(numAllPairNodes, 0);
  std::vector<int> numInArcs(numAllPairNodes, 0);

  // Tracks the nodes that are going to be present in the combined graph
  std::fill(toExplore.begin(), toExplore.end(), false);

  {
    for (int i = 0; i < numAllPairNodes; ++i) {
      std::pair<int, int> indices = OneDToTwoDIndex(i, numNodesFirst);

      if (graphDP1.start[indices.first] && graphDP2.start[indices.second]) {
        if (reachable[i]) {
          toExplore[i] = true;
          newNodes[i] = true;
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

    std::vector<int> arcCrossProductOffset;
    std::pair<std::vector<int>, std::vector<int>> toExploreNumArcs;
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

        const bool epsilonMatch = epsilonMatched[TwoDToOneDIndex(
            nodePair.first, nodePair.second, numNodesFirst)];

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
          if (graphDP1.olabels[firstArcIdx] != epsilon) {
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
  // Step 3: Generate offsets for nodes and arcs in combined graph
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
  // Step 4: Generate nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////
  std::fill(toExplore.begin(), toExplore.end(), false);
  std::vector<int> newNodesVisited(numAllPairNodes, false);

  {
    for (int i = 0; i < numAllPairNodes; ++i) {
      std::pair<int, int> indices = OneDToTwoDIndex(i, numNodesFirst);

      if (graphDP1.start[indices.first] && graphDP2.start[indices.second]) {
        if (reachable[i]) {
          toExplore[i] = true;
          newNodesVisited[i] = true;
          newGraphDP.start[newNodesOffset[i]] = true;
          newGraphDP.accept[newNodesOffset[i]] =
              graphDP1.accept[indices.first] && graphDP2.accept[indices.second];
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

    std::vector<int> arcCrossProductOffset;
    std::pair<std::vector<int>, std::vector<int>> toExploreNumArcs;
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

        const bool epsilonMatch = epsilonMatched[TwoDToOneDIndex(
            srcNodePair.first, srcNodePair.second, numNodesFirst)];

        // Does this node pair match?
        if (checkArcPair &&
            (graphDP1.olabels[firstArcIdx] == graphDP2.ilabels[secondArcIdx])) {
          std::pair<int, int> dstNodePair = std::make_pair(
              graphDP1.dstNodes[firstArcIdx], graphDP2.dstNodes[secondArcIdx]);

          const int dstIdx = TwoDToOneDIndex(
              dstNodePair.first, dstNodePair.second, numNodesFirst);
          const int curIdx = TwoDToOneDIndex(
              srcNodePair.first, srcNodePair.second, numNodesFirst);

          std::pair<bool, bool> dstNodeStartAndAccept =
              getStartAndAccept(graphDP1, graphDP2, dstNodePair);

          // We track if any two arcs outgoing from this node pair match
          // on epsilon. We record if they do.
          if (graphDP1.olabels[firstArcIdx] != epsilon) {
            generateCombinedGraphNodesAndArcs(
                dstIdx,
                curIdx,
                std::make_pair(firstArcIdx, secondArcIdx),
                dstNodeStartAndAccept,
                reachable,
                newNodesOffset,
                newNodesVisited,
                toExplore,
                gradInfo,
                newGraphDP,
                graphDP1.ilabels[firstArcIdx],
                graphDP2.olabels[secondArcIdx],
                graphDP1.weights[firstArcIdx] + graphDP2.weights[secondArcIdx]);
          }
        }

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

          std::pair<bool, bool> dstNodeStartAndAccept =
              getStartAndAccept(graphDP1, graphDP2, dstNodePair);

          generateCombinedGraphNodesAndArcs(
              dstIdx,
              curIdx,
              std::make_pair(firstArcIdx, -1),
              dstNodeStartAndAccept,
              reachable,
              newNodesOffset,
              newNodesVisited,
              toExplore,
              gradInfo,
              newGraphDP,
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

          std::pair<bool, bool> dstNodeStartAndAccept =
              getStartAndAccept(graphDP1, graphDP2, dstNodePair);

          generateCombinedGraphNodesAndArcs(
              dstIdx,
              curIdx,
              std::make_pair(-1, secondArcIdx),
              dstNodeStartAndAccept,
              reachable,
              newNodesOffset,
              newNodesVisited,
              toExplore,
              gradInfo,
              newGraphDP,
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
