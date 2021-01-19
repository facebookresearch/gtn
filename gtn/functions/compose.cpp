/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
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

/*
 * Find any state in the new composed graph which can reach
 * an accepting state.
 *
 * This is accomplished by iteatively following backwards pairwise arc paths
 * from the first and second graphs, where the arc paths have the invariant
 * that, for a particular arc pair, the olabel for the first arc == the ilabel
 * for the second arc.
 */
auto findReachable(
    const Graph& first,
    const Graph& second,
    std::shared_ptr<ArcMatcher> matcher) {
  std::vector<bool> reachable(first.numNodes() * second.numNodes(), false);
  std::queue<std::pair<int, int>> toExplore;
  // toExplore -- add accepting node pairs
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
    // Iterate through arcs that end with the curr node - the first arc's olabel
    // is the same as the second arc's ilabel per the matcher
    while (matcher->hasNext()) {
      std::tie(i, j) = matcher->next(); // arcs ending with curr
      epsilon_matched |= (first.olabel(i) == epsilon);
      // Starting nodes for i and j arcs
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
  // Prospective new dest node in the composed graph. Ignore if we can't get to
  // an accept state.
  auto idx = toIndex(dstNodes.first, dstNodes.second, first);
  if (reachable[idx]) {
    // Build the node - val of -1 --> uninitialized
    if (newNodes[idx] < 0) {
      newNodes[idx] = ngraph.addNode(
          first.isStart(dstNodes.first) && second.isStart(dstNodes.second),
          first.isAccept(dstNodes.first) && second.isAccept(dstNodes.second));
      // Explore forward
      toExplore.emplace(dstNodes.first, dstNodes.second);
    }
    auto newarc =
        ngraph.addArc(currNode, newNodes[idx], ilabel, olabel, weight);
  }
  return reachable[idx];
}

/*
 * For epsilon arcs in either the first or second graph: an epsilon output for
 * some arc a in the first graph maps to a (ilabel(a) --> [second graph olabel])
 * arc in the composed graph, and an epsilon input for some arc a second graph
 * maps to a ([first graph ilabel] --> olabel(a)) arc in the composed graph.
 *
 * The weight of the new arc is equal to the non-epsilon arc's weight.
 */
void addEpsilonReachableNodes(
    bool secondOrFirst,
    const Graph& first,
    const Graph& second,
    int currNode, // in the composed graph
    const std::pair<int, int>& nodePair, // in the input graphs
    const std::vector<bool>& reachable,
    std::queue<std::pair<int, int>>& toExplore,
    std::vector<int>& newNodes,
    Graph& ngraph,
    std::vector<std::pair<int, int>>& gradInfo) {
  auto edges =
      secondOrFirst ? second.out(nodePair.second) : first.out(nodePair.first);
  // If epsilon is the output of an arc in the first graph's current node,
  // add an edge from the current node in the composed graph that takes epsilon
  // --> the second graph's olabel; if the second graph contains an input
  // epsilon, add an edge that takes the first graph's ilabel --> epsilon.
  // Traverse the corresponding edge in either graph and explore it forward
  // since the subgraph reachable from it is valid in the composed graph
  for (auto i : edges) {
    auto label = secondOrFirst ? second.ilabel(i) : first.olabel(i);
    auto isSorted =
        secondOrFirst ? second.ilabelSorted() : first.olabelSorted();
    if (label != epsilon) {
      if (isSorted) {
        // epsilon < 0 - can early-stop since we've reached a non-epsilon node
        // which will appear after epsilons in the sorted order
        break;
      } else {
        continue; // might find a future epsilon
      }
    }

    // The destination node in the composed graph
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
      // The edge with this corresponding gradInfo index in the composed graph
      // corresponds to index i in the first/second graph. Wlog, if advancing
      // along an epsilon edge in the first graph, shouldn't have the gradient
      // corresponding to the resulting edge in the composed graph applied to it
      // at backwards time.
      if (secondOrFirst) {
        gradInfo.emplace_back(-1, i);
      } else {
        gradInfo.emplace_back(i, -1);
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
  // Flat representation of nodes in both graphs, indexed using toIndex
  std::vector<int> newNodes(first.numNodes() * second.numNodes(), -1);
  std::queue<std::pair<int, int>> toExplore;
  // Compile starting nodes that are reachable. If any pairs of reachable start
  // nodes in the input graph are also both accept nodes, make these accept
  // nodes in the composed graph.
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

  // The index of a particlar pair entry in gradInfo corresponds to an arc in
  // the composed graph - at gradient computation time, this facilitates
  // efficiently mapping an arc in the composed graph to the corresponding arcs
  // in the first and second graphs
  std::vector<std::pair<int, int>> gradInfo;
  // Explore the graph starting from the collection of start nodes
  while (!toExplore.empty()) {
    auto curr = toExplore.front();
    toExplore.pop();
    // A node in the composed graph
    auto currNode = newNodes[toIndex(curr.first, curr.second, first)];
    int i, j;
    matcher->match(curr.first, curr.second);
    // Each pair of nodes in the initial graph may have multiple outgoing arcs
    // that should be combined in the composed graph
    while (matcher->hasNext()) {
      // The matcher invariant remains: arc i's olabel (from the first graph) is
      // arc j's ilabel (from the second graph)
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
        // Arcs remember where they came from for easy gradient computation.
        gradInfo.emplace_back(i, j);
      }
    }
    // Check for output epsilons in the first graph
    addEpsilonReachableNodes(
        false,
        first,
        second,
        currNode, // in the composed graph
        curr, // in the input graphs
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
        currNode, // in the composed graph
        curr, // in the input graphs
        reachable,
        toExplore,
        newNodes,
        ngraph,
        gradInfo);
  }

  /*
   * Here we assume deltas is the output (e.g. ngraph) and we know where
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
