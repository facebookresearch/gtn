/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <queue>

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
      gradInfo.emplace_back(-1, i);
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

  // One value per node
  // i-th value corresponds to i-th node
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
// Exclusive prefix sum
void sumScan(std::vector<int>& input) {
  int sum = 0;
  for (auto i : input.size()) {
    auto count = input[i];
    input[i] = sum;
    sum += count;
  }
  input.push_back(sum);
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

inline bool checkAnyTrue(const std::vector<bool>& flags) {
  for (auto i : flags) {
    if (i == true) {
      return i;
    }
  }
  return false;
}

// Map thread id to corresponding node and edge pair
// Search to find which node pair this tid will fall into
// Linear search for now (edgeMullOffset is sorted by definition)
std::pair<std::pair<int, int>, std::pair<int, int>> computeNodeAndEdgePair(
    int tid,
    const std::vector<int>& edgeMullOffset,
    const std::vector<std::pair<int, int>>& toExploreNodePair) {
  std::pair<int, int> nodePair;
  std::pair<int, int> edgePair;
  for (size_t i = 0; i < edgeMullOffset.size() - 1; ++i) {
    const int lVal = edgeMullOffset[i];
    const int rVal = edgeMullOffset[i + 1];

    if ((lVal <= tid) && (tid < rVal)) {
      nodePair = toExploreNodePair[i];
      const int numEdges = edgeMullOffset[i + 1] - edgeMullOffset[i];
      edgePair = OneDToTwoDIndex(numEdges, toExploreNumEdges[i].first);
    }
  }

  return std::make_pair(nodePair, edgePair);
}
} // namespace

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

// Convert from AOS to SOA
GraphDataParaellel convertToDataParallel(const Graph& graph) {
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
  prefixSumScan(graphDP.inArcOffset);
  prefixSumScan(graphDP.outArcOffset);

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

Graph compose(const Graph& first, const Graph& second) {
  GraphDataParallel graphDP1, graphDP2;
  // Convert from AOS to SOA
  graphDP1 = convertToDataParallel(first);
  graphDP2 = convertToDataParallel(second);

  // A first attempt at data parallel findReachable
  std::vector<bool> toExplore(first.numNodes() * second.numNodes(), false);
  std::vector<bool> reachable(first.numNodes() * second.numNodes(), false);
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

    // Calculate number of threads we have to spawn
    std::vector<std::pair<int.int>> toExploreNumEdge(toExploreNodePair.size());
    std::vector<int> edgeMulOffset(toExploreNodePair.size());

    // No dependence between iterations
    for (int i = 0; i < toExploreNodePair.size(); ++i) {
      int node = toExploreNodePair[i].first;
      const int numEdgesFirst =
          graphDP1.inArcOffset[node + 1] - graphDP1.inArcOffset[node];

      node = toExploreNodePair[i].second;
      const int numEdgesSecond =
          graphDP2.inArcOffset[node + 1] - graphDP2.inArcOffset[node];

      toExploreNumEdge[i] = std::make_pair(numEdgesFirst, numEdgesSecond);
      edgeMulOffset[i] = numEdgesFirst * numEdgesSecond;
    }

    sumScan(edgeMullOffset);
    assert(!edgeMulOffset.empty());
    const int totalEdges = edgeMullOffset[edgeMullOffset.size() - 1];

    // No dependence between iterations. tid is thread-id
    // Only do non epsilon case for this kernel
    for (int tid = 0; tid < totalEdges; ++tid) {
      // Node pair
      std::pair<int, int> nodePair;
      std::pair<int, int> edgePair;

      // Map tid to corresponding node and edge pair
      // Search to find which node pair this tid will fall into
      // Linear search for now (edgeMullOffset is sorted by definition)
      std::vector<int, int> nodePair;
      std::vector<int, int> edgePair;
      std::tie(nodePair, edgePair) =
          computeNodeAndEdgePair(tid, edgeMullOffset, toExploreNodePair);

      // Does this node pair match?
      if (graphDP1.olabels[edgePair.first] ==
          graphDP2.ilabels[edgePair.second]) {
        const int idx = TwoDToOneDIndex(
            graphDP1.srcNodes[edgePair.first],
            graphDP2.srcNodes[edgePair.second],
            numNodesFirst);
        if (!reachable[idx]) {
          toExplore[idx] = true;
        }
        reachable[idx] = true;

        if (graphDP1.olabels[edgePair.first] == epsilon) {
          epsilonMatched[TwoDToOneDIndex(
              nodePair.first, nodePair.second, numNodesFirst)] = true;
        }
      }
    }

    // No dependence between iterations. tid is thread-id
    // Do epsilon match case in this kernel launch
    for (int tid = 0; tid < totalEdges; ++tid) {
      // Node pair
      std::pair<int, int> nodePair;
      std::pair<int, int> edgePair;

      // Map tid to corresponding node and edge pair
      // Search to find which node pair this tid will fall into
      // Linear search for now (edgeMullOffset is sorted by definition)
      std::vector<int, int> nodePair;
      std::vector<int, int> edgePair;
      std::tie(nodePair, edgePair) =
          computeNodeAndEdgePair(tid, edgeMullOffset, toExploreNodePair);

      const bool matched = epsilonMatched[TwoDToOneDIndex(
          nodePair.first, nodePair.second, numNodesFirst)];
      // Note that when this loop body is threaded - multiple threads will
      // write to the same location. They will write the same value so
      // this is fine
      if (graphDP1.olabels[edgePair.first] == epsilon && !matched) {
        const int idx = TwoDToOneDIndex(
            graphDP1.srcNodes[edgePair.first], nodePair.second, numNodesFirst);
        if (!reachable[idx]) {
          toExplore[idx] = true;
        }
        reachable[idx] = true;
      }

      // Note that when this loop body is threaded - multiple threads will
      // write to the same location. They will write the same value so
      // this is fine
      if (graphDP1.ilabels[edgePair.second] == epsilon && !matched) {
        const int idx = TwoDToOneDIndex(
            nodePair.first, graphDP2.srcNodes[edgePair.second], numNodesFirst);
        if (!reachable[idx]) {
          toExplore[idx] = true;
        }
        reachable[idx] = true;
      }
    }
  } // end while for findReachable


  // Begin first pass to generate metadata for valid nodes and edges. This
  // is needed before we can generate the nodes and edges themselves.
  std::fill(toExplore.begin(), toExplore.end(), false);

  // This vector tracks the nodes that are going to be present in the combined
  // graph
  std::vector<bool> newNodes(first.numNodes() * second.numNodes(), false);
  std::vector<int> numOutArcs(first.numNodes() * second.numNodes(), 0);

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
    // std::fill(epsilonMatched.begin(), epsilonMatched.end(), false);

    // Calculate number of threads we have to spawn
    std::vector<std::pair<int.int>> toExploreNumEdge(toExploreNodePair.size());
    std::vector<int> edgeMulOffset(toExploreNodePair.size());

    // No dependence between iterations
    for (int i = 0; i < toExploreNodePair.size(); ++i) {
      int node = toExploreNodePair[i].first;
      const int numEdgesFirst =
          graphDP1.outArcOffset[node + 1] - graphDP1.outArcOffset[node];

      node = toExploreNodePair[i].second;
      const int numEdgesSecond =
          graphDP2.inArcOffset[node + 1] - graphDP2.outArcOffset[node];

      toExploreNumEdge[i] = std::make_pair(numEdgesFirst, numEdgesSecond);
      edgeMulOffset[i] = numEdgesFirst * numEdgesSecond;
    }

    sumScan(edgeMullOffset);
    assert(!edgeMulOffset.empty());
    const int totalEdges = edgeMullOffset[edgeMullOffset.size() - 1];

    for (int tid = 0; tid < totalEdges; ++tid) {
      // Node pair
      std::pair<int, int> nodePair;
      std::pair<int, int> edgePair;

      // Map tid to corresponding node and edge pair
      // Search to find which node pair this tid will fall into
      // Linear search for now (edgeMullOffset is sorted by definition)
      std::vector<int, int> nodePair;
      std::vector<int, int> edgePair;
      std::tie(nodePair, edgePair) =
          computeNodeAndEdgePair(tid, edgeMullOffset, toExploreNodePair);

      // Does this node pair match?
      if (graphDP1.olabels[edgePair.first] ==
          graphDP2.ilabels[edgePair.second]) {
        const int dstIdx = TwoDToOneDIndex(
            graphDP1.dstNodes[edgePair.first],
            graphDP2.dstNodes[edgePair.second],
            numNodesFirst);
        const int curIdx = TwoDToOneDIndex(
            nodePair.first, nodePair.second, numNodesFirst);
        if (reachable[dstIdx]) {
          if (!newNodes[dstIdx]) {
            newNodes[idx] = true;
            toExplore[idx] = true;
          }
          // This is an atomic increment
          numOutArcs[curIdx]++;
        }
      }
    }
  }
}

} // namespace dataparallel
} // namespace detail
} // namespace gtn
