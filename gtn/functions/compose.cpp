/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <queue>
#include <tuple>

#include "gtn/functions/compose.h"

namespace gtn {
namespace detail {
namespace {

inline size_t toIndex(int n1, int n2, const Graph& g) {
  return n1 + g.numNodes() * n2;
}

/* Check reachability via edges with epsilon labels */
void epsilonReachable(
    bool secondOrFirst,
    const Graph& first,
    const Graph& second,
    const std::pair<int, int>& nodePair,
    std::vector<bool>& reachable,
    std::queue<std::pair<int, int>>& toExplore) {
  auto edges =
      secondOrFirst ? second.in(nodePair.second) : first.in(nodePair.first);

  for (auto i : edges) {
    auto label = secondOrFirst ? second.ilabel(i) : first.olabel(i);
    auto isSorted =
        secondOrFirst ? second.ilabelSorted() : first.olabelSorted();
    if (label != epsilon) {
      if (isSorted) {
        break;
      } else {
        continue;
      }
    }
    auto un = secondOrFirst ? second.srcNode(i) : first.srcNode(i);
    auto idx = secondOrFirst ? toIndex(nodePair.first, un, first)
                             : toIndex(un, nodePair.second, first);
    if (!reachable[idx]) {
      // If we haven't seen this state before, explore it.
      secondOrFirst ? toExplore.emplace(nodePair.first, un)
                    : toExplore.emplace(un, nodePair.second);
    }
    reachable[idx] = true;
  }
}

/* Find any state in the new composed graph which can reach
 * an accepting state. */
auto findReachable(
    const Graph& first,
    const Graph& second,
    std::shared_ptr<ArcMatcher> matcher) {
  std::vector<bool> reachable(first.numNodes() * second.numNodes(), false);
  std::queue<std::pair<int, int>> toExplore;
  for (auto f : first.accept()) {
    for (auto s : second.accept()) {
      toExplore.emplace(f, s);
      reachable[toIndex(f, s, first)] = true;
    }
  }

  while (!toExplore.empty()) {
    auto curr = toExplore.front();
    toExplore.pop();

    bool epsilon_matched = false;
    matcher->match(curr.first, curr.second, true);
    int i, j;
    while (matcher->hasNext()) {
      std::tie(i, j) = matcher->next();
      epsilon_matched |= (first.olabel(i) == epsilon);
      auto un1 = first.srcNode(i);
      auto un2 = second.srcNode(j);
      auto idx = toIndex(un1, un2, first);
      if (!reachable[idx]) {
        // If we haven't seen this state before, explore it.
        toExplore.emplace(un1, un2);
      }
      reachable[idx] = true;
    }
    if (!epsilon_matched) {
      // Check for reachable node via output epsilon first graph
      epsilonReachable(false, first, second, curr, reachable, toExplore);
      // Check for reachable node via input epsilon in second graph
      epsilonReachable(true, first, second, curr, reachable, toExplore);
    }
  }
  return reachable;
}

/* Add a node and arc to the new graph if it is reachable.
 * Returns if node is reachable. */
bool addReachableNodeAndArc(
    const Graph& first,
    const Graph& second,
    int currNode,
    const std::pair<int, int>& dstNodes,
    float weight,
    int ilabel,
    int olabel,
    const std::vector<bool>& reachable,
    std::queue<std::pair<int, int>>& toExplore,
    std::vector<int>& newNodes,
    Graph& ngraph) {
  // Ignore if we can't get to an accept state.
  auto idx = toIndex(dstNodes.first, dstNodes.second, first);
  if (reachable[idx]) {
    // Build the node
    if (newNodes[idx] < 0) {
      newNodes[idx] = ngraph.addNode(
          first.isStart(dstNodes.first) && second.isStart(dstNodes.second),
          first.isAccept(dstNodes.first) && second.isAccept(dstNodes.second));
      toExplore.emplace(dstNodes.first, dstNodes.second);
    }
    auto newarc =
        ngraph.addArc(currNode, newNodes[idx], ilabel, olabel, weight);
  }
  return reachable[idx];
}

void addEpsilonReachableNodes(
    bool secondOrFirst,
    const Graph& first,
    const Graph& second,
    int currNode,
    const std::pair<int, int>& nodePair,
    const std::vector<bool>& reachable,
    std::queue<std::pair<int, int>>& toExplore,
    std::vector<int>& newNodes,
    Graph& ngraph,
    std::vector<std::pair<int, int>>& gradInfo) {
  auto edges =
      secondOrFirst ? second.out(nodePair.second) : first.out(nodePair.first);
  for (auto i : edges) {
    auto label = secondOrFirst ? second.ilabel(i) : first.olabel(i);
    auto isSorted =
        secondOrFirst ? second.ilabelSorted() : first.olabelSorted();
    if (label != epsilon) {
      if (isSorted) {
        // epsilon < 0
        break;
      } else {
        continue;
      }
    }

    bool isReachable = addReachableNodeAndArc(
        first,
        second,
        currNode,
        std::make_pair(
            secondOrFirst ? nodePair.first : first.dstNode(i),
            secondOrFirst ? second.dstNode(i) : nodePair.second),
        secondOrFirst ? second.weight(i) : first.weight(i),
        secondOrFirst ? epsilon : first.ilabel(i),
        secondOrFirst ? second.olabel(i) : epsilon,
        reachable,
        toExplore,
        newNodes,
        ngraph);

    if (isReachable) {
      if (secondOrFirst) {
        gradInfo.emplace_back(i, -1);
      } else {
        gradInfo.emplace_back(-1, i);
      }
    }
  }
}
} // namespace

void UnsortedMatcher::match(int lnode, int rnode, bool matchIn /* = false*/) {
  auto& lv = matchIn ? g1_.in(lnode) : g1_.out(lnode);
  auto& rv = matchIn ? g2_.in(rnode) : g2_.out(rnode);
  lIt_ = lv.begin();
  lItEnd_ = lv.end();
  rItBegin_ = rIt_ = rv.begin();
  rItEnd_ = rv.end();
}

bool UnsortedMatcher::hasNext() {
  for (; lIt_ != lItEnd_; ++lIt_) {
    for (; rIt_ != rItEnd_; ++rIt_) {
      if (g1_.olabel(*lIt_) == g2_.ilabel(*rIt_)) {
        return true;
      }
    }
    rIt_ = rItBegin_;
  }
  return false;
}

std::pair<int, int> UnsortedMatcher::next() {
  return std::make_pair(*lIt_, *rIt_++);
}

SinglySortedMatcher::SinglySortedMatcher(
    const Graph& g1,
    const Graph& g2,
    bool searchG1 /* = false */)
    : g1_(g1), g2_(g2), searchG1_(searchG1) {}

void SinglySortedMatcher::match(
    int lnode,
    int rnode,
    bool matchIn /* = false */) {
  auto& lv = matchIn ? g1_.in(lnode) : g1_.out(lnode);
  auto& rv = matchIn ? g2_.in(rnode) : g2_.out(rnode);

  searchItBegin_ = searchIt_ = lv.begin();
  searchItEnd_ = lv.end();
  queryIt_ = rv.begin();
  queryItEnd_ = rv.end();

  if (!searchG1_) {
    searchItBegin_ = queryIt_;
    std::swap(queryIt_, searchIt_);
    std::swap(queryItEnd_, searchItEnd_);
  }
}

bool SinglySortedMatcher::hasNext() {
  if (queryIt_ == queryItEnd_) {
    return false;
  }
  if (searchIt_ != searchItEnd_) {
    auto ql = searchG1_ ? g2_.ilabel(*queryIt_) : g1_.olabel(*queryIt_);
    auto sl = searchG1_ ? g1_.olabel(*searchIt_) : g2_.ilabel(*searchIt_);
    if (ql == sl) {
      return true;
    }
  }
  if (searchIt_ != searchItBegin_) {
    // Not at the start of the search
    ++queryIt_;
  }

  // Update the query pointer and the start of the search range pointer
  for (; queryIt_ != queryItEnd_; ++queryIt_) {
    auto ql = searchG1_ ? g2_.ilabel(*queryIt_) : g1_.olabel(*queryIt_);
    // Set the comparison function appropriately
    auto comparisonFn = [this](int arc, int val) {
      return searchG1_ ? g1_.olabel(arc) < val : g2_.ilabel(arc) < val;
    };
    searchIt_ =
        std::lower_bound(searchItBegin_, searchItEnd_, ql, comparisonFn);

    if (searchIt_ == searchItEnd_) {
      continue;
    }

    auto sl = searchG1_ ? g1_.olabel(*searchIt_) : g2_.ilabel(*searchIt_);
    if (sl == ql) {
      return true;
    }
  }
  return false;
}

std::pair<int, int> SinglySortedMatcher::next() {
  if (searchG1_) {
    return std::make_pair(*searchIt_++, *queryIt_);
  } else {
    return std::make_pair(*queryIt_, *searchIt_++);
  }
}

void DoublySortedMatcher::match(
    int lnode,
    int rnode,
    bool matchIn /* = false */) {
  auto& lv = matchIn ? g1_.in(lnode) : g1_.out(lnode);
  auto& rv = matchIn ? g2_.in(rnode) : g2_.out(rnode);

  searchItBegin_ = searchIt_ = lv.begin();
  searchItEnd_ = lv.end();
  queryIt_ = rv.begin();
  queryItEnd_ = rv.end();

  searchG1_ = lv.size() > rv.size();
  if (!searchG1_) {
    searchItBegin_ = queryIt_;
    std::swap(queryIt_, searchIt_);
    std::swap(queryItEnd_, searchItEnd_);
  }
}

bool DoublySortedMatcher::hasNext() {
  if (queryIt_ == queryItEnd_) {
    return false;
  }
  if (searchIt_ != searchItEnd_) {
    auto ql = searchG1_ ? g2_.ilabel(*queryIt_) : g1_.olabel(*queryIt_);
    auto sl = searchG1_ ? g1_.olabel(*searchIt_) : g2_.ilabel(*searchIt_);
    if (ql == sl) {
      return true;
    }
  }
  if (searchIt_ != searchItBegin_) {
    // Not at the start of the search
    ++queryIt_;
  }

  // Update the query pointer and the start of the search range pointer
  for (; queryIt_ != queryItEnd_; ++queryIt_) {
    auto ql = searchG1_ ? g2_.ilabel(*queryIt_) : g1_.olabel(*queryIt_);

    // Set the comparison function appropriately
    auto comparisonFn = [this](int arc, int val) {
      return searchG1_ ? g1_.olabel(arc) < val : g2_.ilabel(arc) < val;
    };
    // Allowed because the query vector is sorted.
    searchItBegin_ =
        std::lower_bound(searchItBegin_, searchItEnd_, ql, comparisonFn);
    if (searchItBegin_ == searchItEnd_) {
      return false;
    }

    auto sl =
        searchG1_ ? g1_.olabel(*searchItBegin_) : g2_.ilabel(*searchItBegin_);
    if (sl == ql) {
      searchIt_ = searchItBegin_;
      return true;
    }
  }
  return false;
}

std::pair<int, int> DoublySortedMatcher::next() {
  if (searchG1_) {
    return std::make_pair(*searchIt_++, *queryIt_);
  } else {
    return std::make_pair(*queryIt_, *searchIt_++);
  }
}

// Composes two graphs and returns a new graph
Graph compose(
    const Graph& first,
    const Graph& second,
    std::shared_ptr<ArcMatcher> matcher) {
  // Compute reachable nodes from any accept state in the new graph
  auto reachable = findReachable(first, second, matcher);

  // Compose the graphs
  Graph ngraph(nullptr, {first, second});
  std::vector<int> newNodes(first.numNodes() * second.numNodes(), -1);
  std::queue<std::pair<int, int>> toExplore;
  for (auto s1 : first.start()) {
    for (auto s2 : second.start()) {
      auto idx = toIndex(s1, s2, first);
      if (reachable[idx]) {
        newNodes[idx] =
            ngraph.addNode(true, first.isAccept(s1) && second.isAccept(s2));
        toExplore.emplace(s1, s2);
      }
    }
  }
  std::vector<std::pair<int, int>> gradInfo;
  while (!toExplore.empty()) {
    auto curr = toExplore.front();
    toExplore.pop();
    auto currNode = newNodes[toIndex(curr.first, curr.second, first)];
    int i, j;
    matcher->match(curr.first, curr.second);
    while (matcher->hasNext()) {
      std::tie(i, j) = matcher->next();

      bool isReachable = addReachableNodeAndArc(
          first,
          second,
          currNode,
          std::make_pair(first.dstNode(i), second.dstNode(j)),
          first.weight(i) + second.weight(j),
          first.ilabel(i),
          second.olabel(j),
          reachable,
          toExplore,
          newNodes,
          ngraph);

      if (isReachable) {
        // Arcs remember where they came from for
        // easy gradient computation.
        gradInfo.emplace_back(i, j);
      }
    }
    // Check for output epsilons in the first graph
    addEpsilonReachableNodes(
        false,
        first,
        second,
        currNode,
        curr,
        reachable,
        toExplore,
        newNodes,
        ngraph,
        gradInfo);
    // Check out input epsilons in the second graph
    addEpsilonReachableNodes(
        true,
        first,
        second,
        currNode,
        curr,
        reachable,
        toExplore,
        newNodes,
        ngraph,
        gradInfo);
  }

  /* Here we assume deltas is the output (e.g. ngraph) and we know where
   * each arc came from. This makes it possible to disambiguate two arcs in the
   * composed graph with the same label and the same src and destination nodes.
   */
  auto gradFunc = [gradInfo = std::move(gradInfo)](
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

  ngraph.setGradFunc(std::move(gradFunc));
  return ngraph;
}

} // namespace detail
} // namespace gtn

namespace gtn {
namespace detail {
namespace dataparallel {
// Change AOS to SOA
struct GraphDataParallel {
  std::vector<int> accept;
  std::vector<int> start;

  // Has one more element than the number of nodes
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

// Remove all nodes that have 0 in and out arcs - these nodes are unreachable
// This is essentially a stream compact
std::tuple<std::vector<int>, std::vector<int>> removeUnreachableNodes(
    const std::vector<int>& numInArcs,
    const std::vector<int>& numOutArcs) {
  std::vector<int> outputInArcs;
  std::vector<int> outputOutArcs;

  assert(numInArcs.size() == numOutArcs.size());

  for (size_t i = 0; i < numInArcs.size(); ++i) {
    if ((numInArcs[i] != 0) || (numOutArcs[i] != 0)) {
      outputInArcs.push_back(numInArcs[i]);
      outputOutArcs.push_back(numOutArcs[i]);
    }
  }

  return std::make_pair(outputInArcs, outputOutArcs);
}

// TODO: Duplicate - should be removed
inline int TwoDToOneDIndex(int n1, int n2, int n1Extent) {
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
std::
    tuple<bool, std::pair<int, int>, std::pair<int, int>, std::pair<bool, bool>>
    computeNodeAndArcPair(
        int tid,
        const std::vector<int>& arcCrossProductOffset,
        const std::vector<int>& toExploreNumArcs,
        const std::vector<std::pair<int, int>>& toExploreNodePair) {
  std::pair<int, int> nodePair;
  std::pair<int, int> arcPair;
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
      const int numArcs =
          arcCrossProductOffset[i + 1] - arcCrossProductOffset[i];
      arcPair = OneDToTwoDIndex(numArcs, toExploreNumArcs[i].first);

      // The range of idx is from
      // [0, toExploreNumArcs[i].first * toExploreNumArcs[i].second)
      const int idx = tid - lVal;

      // We map the tids to 2D grid where the
      // x-axis is toExploreNumArcs[i].second (row)
      // y-axis is toExploreNumArcs[i].first (column)

      // Pick the tids from the first column since we need only one
      // tid per arc of the node from the first graph to check for
      // epsilon
      if (idx % toExploreNumArcs[i].second == 0) {
        checkEpsilonArcPair.first = true;
      }

      // Pick the tids from the first row since we need only one
      // tid per arc of the node from the first graph to check for
      // epsilon
      if (idx < toExploreNumArcs[i].second) {
        checkEpsilonArcPair.second = true;
      }

      break;
    }
  }

  return std::make_tuple(isValid, nodePair, arcPair, checkEpsilonArcPair);
}

// Takes a pair of nodes, where each member of pair comes from a different
// graph and calculate a vector of number of arcs in the cross product of
// arcs outgoing from each pair.
// This should be a kernel call
std::tuple<std::vector<int>, std::vector<int>> calculateArcCrossProductOffset(
    const std::vector<int>& toExploreNodePair,
    const GraphDataParallel& graphDP1,
    const GraphDataParallel& graphDP2,
    bool inOrOutArc) {
  std::vector<std::pair<int, int>> toExploreNumArcs(toExploreNodePair.size());
  std::vector<int> arcCrossProductOffset(toExploreNodePair.size());

  // No dependence between iterations
  for (size_t i = 0; i < toExploreNodePair.size(); ++i) {
    int node = toExploreNodePair[i].first;
    const int numArcsFirst = inOrOutArc
        ? graphDP1.inArcOffset[node + 1] - graphDP1.inArcOffset[node]
        : graphDP1.outArcOffset[node + 1] - graphDP1.outArcOffset[node];

    node = toExploreNodePair[i].second;
    const int numArcsSecond = inOrOutArc
        ? graphDP2.inArcOffset[node + 1] - graphDP2.inArcOffset[node]
        : graphDP2.outArcOffset[node + 1] - graphDP2.outArcOffset[node];

    toExploreNumArcs[i] = std::make_pair(numArcsFirst, numArcsSecond);
    arcCrossProductOffset[i] = numArcsFirst * numArcsSecond;
  }

  return std::make_tuple(arcCrossProductOffset, toExploreNumArcs);
}

// This function needs to be thread safe since multiple threads can
// can call it and they will overlap on curIdx and dstIdx
void calculateNumArcsAndNodesToExplore(int curIdx,
                                       int dstIdx,
                                       const std::vector<bool>& reachable,
                                       std::vector<bool>& newNodes;
                                       std::vector<bool> & toExplore,
                                       std::vector<int> numOutArcs,
                                       std::vector<int> numInArcs) {
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
    int ilabel,
    int olabel,
    int weight) {
  if (reachable[dstIdx]) {
    // Atomic test and set for newNodesVisited
    if (!newNodesVisited[dstIdx]) {
      newNodesVisited[dstIdx] = true;
      toExplore[dstIdx] = true;
    }

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

} // namespace

// Convert from AOS to SOA
GraphDataParallel convertToDataParallel(const Graph& graph) {
  GraphDataParallel graphDP;

  graphDP.inArcOffset.resize(graph.numNodes());
  graphDP.outArcOffset.resize(graph.numNodes());

  graphDP.inArcs.resize(graph.numArcs());
  graphDP.outArcs.resize(graph.numArcs());

  graphDP.ilabels.resize(graph.numArcs());
  graphDP.olabels.resize(graph.numArcs());
  graphDP.srcNodes.resize(graph.numArcs());
  graphDP.dstNodes.resize(graph.numArcs());
  graphDP.weights.resize(graph.numArcs());

  for (auto i : graph.start()) {
    graphDP.start.push_back(i);
  }

  for (auto i : graph.accept()) {
    graphDP.accept.push_back(i);
  }

  for (int i = 0; i < graph.numNodes(); ++i) {
    graphDP.inArcOffset[i] = graph.numIn(i);
    graphDP.outArcOffset[i] = graph.numOut(i);
  }

  // Scan of offsets
  const int totalInArcs = prefixSumScan(graphDP.inArcOffset, false);
  const int totalOutArcs = prefixSumScan(graphDP.outArcOffset, false);
  assert(totalInArcs == totalOutArcs);

  for (int i = 0; i < graph.numNodes(); ++i) {
    int offset = graphDP.outArcOffset[i];

    for (auto j : graph.out(i)) {
      graphDP1.outArcs[offset] = j;
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
      graphDP.inArcs[offset] = j;
      offset++;
    }
  }

  return graphDP;
}

// Convert from SOA to AOS
Graph convertFromDataParallel(const GraphDataParallel& graph) {
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
  std::vector<bool> epsilonMatched(first.numNodes() * second.numNodes(), false);

  const int numNodesFirst = first.numNodes();

  for (auto f : graphDP1.accept) {
    for (auto s : graphDP2.accept) {
      toExplore[TwoDToOneDIndex(f, s, numNodesFirst)] = true;
    }
  }

  // This is the outer control loop that would spawn DP kernels
  while (checkAnyTrue(toExplore)) {
    // Convert bits set in toExplore to node pairs
    auto toExploreNodePair = convertToNodePair(toExplore, numNodesFirst);

    // Reset so pristine state for next frontier to explore
    // No dependence between iterations
    std::fill(toExplore.begin(), toExplore.end(), false);
    std::fill(epsilonMatched.begin(), epsilonMatched.end(), false);

    std::vector<int> arcCrossProductOffset;
    std::vector<int> toExploreNumArcs;
    std::tie(arcCrossProductOffset, toExploreNumArcs) =
        calculateArcCrossProductOffset(
            toExploreNodePair, graphDP1, graphDP2, true);

    // If no arcs to process - we are done
    if (arcCrossProductOffset.empty()) {
      break;
    }

    const int totalArcs = prefixSumScan(arcCrossProductOffset, true);

    // No dependence between iterations. tid is thread-id
    // Only do non epsilon case for this kernel
    for (int tid = 0; tid < totalArcs; ++tid) {
      // Node pair
      std::pair<int, int> nodePair;
      std::pair<int, int> arcPair;

      // Map tid to corresponding node and arc pair via search
      std::pair<int, int> nodePair;
      std::pair<int, int> arcPair;
      std::pair<bool, bool> checkEpsilonArcPair;
      bool isValid;
      std::tie(isValid, nodePair, arcPair, checkEpsilonArcPair) =
          computeNodeAndArcPair(
              tid, arcCrossProductOffset, toExploreNumArcs, toExploreNodePair);

      // Does this node pair match?
      if (isValid &&
          (graphDP1.olabels[arcPair.first] ==
           graphDP2.ilabels[arcPair.second])) {
        const int idx = TwoDToOneDIndex(
            graphDP1.srcNodes[arcPair.first],
            graphDP2.srcNodes[arcPair.second],
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
        if (graphDP1.olabels[arcPair.first] == epsilon) {
          epsilonMatched[TwoDToOneDIndex(
              nodePair.first, nodePair.second, numNodesFirst)] = true;
        }
      }
    }

    // No dependence between iterations. tid is thread-id
    // Do epsilon match case in this kernel launch
    for (int tid = 0; tid < totalArcs; ++tid) {
      // Map tid to corresponding node and arc pair via search
      std::pair<int, int> nodePair;
      std::pair<int, int> arcPair;
      std::pair<bool, bool> checkEpsilonArcPair;
      bool isValid;
      std::tie(isValid, nodePair, arcPair, checkEpsilonArcPair) =
          computeNodeAndArcPair(tid, arcCrossProductOffset, toExploreNodePair);

      if (isValid) {
        const bool matched = epsilonMatched[TwoDToOneDIndex(
            nodePair.first, nodePair.second, numNodesFirst)];
        if (!matched && checkEpsilonArcPair.first &&
            (graphDP1.olabels[arcPair.first] == epsilon)) {
          const int idx = TwoDToOneDIndex(
              graphDP1.srcNodes[arcPair.first], nodePair.second, numNodesFirst);
          if (!reachable[idx]) {
            toExplore[idx] = true;
          }
          reachable[idx] = true;
        }

        if (!matched && checkEpsilonArcPair.second &&
            (graphDP2.ilabels[arcPair.second] == epsilon)) {
          const int idx = TwoDToOneDIndex(
              nodePair.first, graphDP2.srcNodes[arcPair.second], numNodesFirst);
          if (!reachable[idx]) {
            toExplore[idx] = true;
          }
          reachable[idx] = true;
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
  // Tracks the nodes that are going to be present in the combined graph
  std::vector<bool> newNodes(first.numNodes() * second.numNodes(), false);
  // Number of in and out arcs per node
  std::vector<int> numOutArcs(first.numNodes() * second.numNodes(), 0);
  std::vector<int> numInArcs(first.numNodes() * second.numNodes(), 0);

  std::fill(toExplore.begin(), toExplore.end(), false);

  for (auto f : graphDP1.start) {
    for (auto s : graphDP2.start) {
      toExplore[TwoDToOneDIndex(f, s, numNodesFirst)] = true;
      newNodes[TwoDToOneDIndex(f, s, numNodesFirst)] = true;
    }
  }

  // This is the outer control loop that would spawn DP kernels
  while (checkAnyTrue(toExplore)) {
    // Convert bits set in toExplore to node pairs
    auto toExploreNodePair = convertToNodePair(toExplore, numNodesFirst);

    // Reset so pristine state for next frontier to epxlore
    // No dependence between iterations
    std::fill(toExplore.begin(), toExplore.end(), false);

    std::vector<int> arcCrossProductOffset;
    std::vector<int> toExploreNumArcs;
    std::tie(arcCrossProductOffset, toExploreNumArcs) =
        calculateArcCrossProductOffset(
            toExploreNodePair, graphDP1, graphDP2, false);

    // If no arcs to process we are done
    if (arcCrossProductOffset.empty() == 0) {
      break;
    }

    const int totalArcs = prefixSumScan(arcCrossProductOffset, true);

    for (int tid = 0; tid < totalArcs; ++tid) {
      // Map tid to corresponding node and arc pair
      // Search to find which node pair this tid will fall into
      std::pair<int, int> nodePair;
      std::pair<int, int> arcPair;
      std::pair<bool, bool> checkEpsilonArcPair;
      bool isValid;
      std::tie(isValid, nodePair, arcPair, checkEpsilonArcPair) =
          computeNodeAndArcPair(tid, arcCrossProductOffset, toExploreNodePair);

      if (isValid) {
        // Does this node pair match?
        if (graphDP1.olabels[arcPair.first] ==
            graphDP2.ilabels[arcPair.second]) {
          const int dstIdx = TwoDToOneDIndex(
              graphDP1.dstNodes[arcPair.first],
              graphDP2.dstNodes[arcPair.second],
              numNodesFirst);
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

        if (checkEpsilonArcPair.first &&
            (graphDP1.olabels[arcPair.first] == epsilon)) {
          const int dstIdx = TwoDToOneDIndex(
              graphDP1.dstNodes[arcPair.first], nodePair.second, numNodesFirst);
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
            (graphDP2.ilabels[arcPair.second] == epsilon)) {
          const int dstIdx = TwoDToOneDIndex(
              nodePair.first, graphDP2.dstNodes[arcPair.second], numNodesFirst);
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
  std::vector<int> newNodesOffset(newNodes.size(), 0);
  for (size_t i = 0; i < newNodes.size(); ++i) {
    if (newNodes[i]) {
      newNodesOffset[i] = 1;
    }
  }

  const int totalNodes = prefixSumScan(newNodesOffset, false);

  // Throw out all nodes with no in or out arcs
  std::tie(newGraphDP.inArcOffset, newGraphDP.outArcOffset) =
      removeUnreachableNodes(numInArcs, numOutArcs);

  // Check that number of nodes match
  assert(totalNodes == newGraphDP.inArcOffset.length());
  assert(newGraphDP.inArcOffset.length() == newGraphDP.outArcOffset.length());

  // Prefix sum to generate offsets
  const int totalInArcs = prefixSumScan(newGraphDP.inArcOffset, false);
  const int totalOutArcs = prefixSumScan(newGraphDP.outArcOffset, false);

  // This is the total number of arcs and they must be equal
  assert(totalInArcs = totaloutArcs);

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
  // Begin first pass to generate metadata for valid nodes and arcs. This
  // is needed before we can generate the nodes and arcs themselves.
  std::fill(toExplore.begin(), toExplore.end(), false);
  std::vector<bool> newNodesVisited(
      first.numNodes() * second.numNodes(), false);

  for (auto f : graphDP1.start) {
    for (auto s : graphDP2.start) {
      toExplore[TwoDToOneDIndex(f, s, numNodesFirst)] = true;
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
    std::vector<int> toExploreNumArcs;
    std::tie(arcCrossProductOffset, toExploreNumArcs) =
        calculateArcCrossProductOffset(
            toExploreNodePair, graphDP1, graphDP2, false);

    // If no arcs to process we are done
    if (arcCrossProductOffset.size() == 0) {
      break;
    }

    const int totalArcs = prefixSumScan(arcCrossProductOffset, true);

    for (int tid = 0; tid < totalArcs; ++tid) {
      // Map tid to corresponding node and arc pair
      // Search to find which node pair this tid will fall into
      std::pair<int, int> nodePair;
      std::pair<int, int> arcPair;
      std::pair<bool, bool> checkEpsilonArcPair;
      bool isValid;
      std::tie(isValid, nodePair, arcPair, checkEpsilonArcPair) =
          computeNodeAndArcPair(tid, arcCrossProductOffset, toExploreNodePair);

      if (isValid) {
        // Does this node pair match?
        if (graphDP1.olabels[arcPair.first] ==
            graphDP2.ilabels[arcPair.second]) {
          const int dstIdx = TwoDToOneDIndex(
              graphDP1.dstNodes[arcPair.first],
              graphDP2.dstNodes[arcPair.second],
              numNodesFirst);
          const int curIdx =
              TwoDToOneDIndex(nodePair.first, nodePair.second, numNodesFirst);

          generateCombinedGraphNodesAndArcs(
              dstIdx,
              curIdx,
              arcPair,
              reachable,
              newNodesOffset,
              newNodesVisited,
              toExplore,
              newGraphDP,
              graphDP1.ilabels[arcPair.first],
              graphDP2.olabels[arcPair.second],
              graphDP1.weights[arcPair.first] +
                  graphDP2.weights[arcPair.second]);
        }

        // The epsilon matches
        if (checkEpsilonArcPair.first &&
            (graphDP1.olabels[arcPair.first] == epsilon)) {
          const int dstIdx = TwoDToOneDIndex(
              graphDP1.dstNodes[arcPair.first], nodePair.second, numNodesFirst);
          const int curIdx =
              TwoDToOneDIndex(nodePair.first, nodePair.second, numNodesFirst);

          generateCombinedGraphNodesAndArcs(
              dstIdx,
              curIdx,
              std::make_pair(arcPair.first, -1),
              reachable,
              newNodesOffset,
              newNodesVisited,
              toExplore,
              newGraphDP,
              graphDP1.ilabels[arcPair.first],
              epsilon,
              graphDP1.weights[arcPair.first]);
        }

        // The epsilon matches
        if (checkEpsilonArcPair.second &&
            (graphDP2.ilabels[arcPair.second] == epsilon)) {
          const int dstIdx = TwoDToOneDIndex(
              nodePair.first, graphDP2.dstNodes[arcPair.second], numNodesFirst);
          const int curIdx =
              TwoDToOneDIndex(nodePair.first, nodePair.second, numNodesFirst);

          generateCombinedGraphNodesAndArcs(
              dstIdx,
              curIdx,
              std::make_pair(-1, arcPair.second),
              reachable,
              newNodesOffset,
              newNodesVisited,
              toExplore,
              newGraphDP,
              epsilon,
              graphDP2.olabels[arcPair.second],
              graphDP2.weights[arcPair.second]);
        }
      }
    }
  }
}

} // namespace dataparallel
} // namespace detail
} // namespace gtn
