#include <algorithm>
#include <queue>

#include "gtn/functions/compose.h"

namespace gtn {
namespace detail {
namespace {

inline size_t toIndex(int n1, int n2, const Graph& g) {
  return n1 + g.numNodes() * n2;
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
      epsilon_matched |= (first.olabel(i) == Graph::epsilon);
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
      for (auto i : first.in(curr.first)) {
        if (first.olabel(i) != Graph::epsilon) {
          if (first.olabelSorted()) {
            // Graph::epsilon < 0
            break;
          } else {
            continue;
          }
        }
        auto un1 = first.srcNode(i);
        auto idx = toIndex(un1, curr.second, first);
        if (!reachable[idx]) {
          // If we haven't seen this state before, explore it.
          toExplore.emplace(un1, curr.second);
        }
        reachable[idx] = true;
      }
    }
    if (!epsilon_matched) {
      for (auto j : second.in(curr.second)) {
        if (second.ilabel(j) != Graph::epsilon) {
          if (second.ilabelSorted()) {
            // Graph::epsilon < 0
            break;
          } else {
            continue;
          }
        }
        auto un2 = second.srcNode(j);
        auto idx = toIndex(curr.first, un2, first);
        if (!reachable[idx]) {
          // If we haven't seen this state before, explore it.
          toExplore.emplace(curr.first, un2);
        }
        reachable[idx] = true;
      }
    }
  }
  return reachable;
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
      auto dn1 = first.dstNode(i);
      auto dn2 = second.dstNode(j);
      // Ignore if we can't get to an accept state.
      auto idx = toIndex(dn1, dn2, first);
      if (!reachable[idx]) {
        continue;
      }
      // Build the node
      if (newNodes[idx] < 0) {
        newNodes[idx] = ngraph.addNode(
            first.isStart(dn1) && second.isStart(dn2),
            first.isAccept(dn1) && second.isAccept(dn2));
        toExplore.emplace(dn1, dn2);
      }
      auto weight = first.weight(i) + second.weight(j);
      auto newarc = ngraph.addArc(
          currNode, newNodes[idx], first.ilabel(i), second.olabel(j), weight);
      // Arcs remember where they came from for
      // easy gradient computation.
      gradInfo.emplace_back(i, j);
    }
    // Check for output epsilons in the first graph
    for (auto i : first.out(curr.first)) {
      if (first.olabel(i) != Graph::epsilon) {
        if (first.olabelSorted()) {
          // Graph::epsilon < 0
          break;
        } else {
          continue;
        }
      }
      // We only advance along the first arc.
      auto dn1 = first.dstNode(i);
      auto dn2 = curr.second;
      auto idx = toIndex(dn1, dn2, first);
      if (!reachable[idx]) {
        continue;
      }
      if (newNodes[idx] < 0) {
        newNodes[idx] = ngraph.addNode(
            first.isStart(dn1) && second.isStart(dn2),
            first.isAccept(dn1) && second.isAccept(dn2));
        toExplore.emplace(dn1, dn2);
      }
      auto weight = first.weight(i);
      auto newarc = ngraph.addArc(
          currNode, newNodes[idx], first.ilabel(i), Graph::epsilon, weight);
      gradInfo.emplace_back(i, -1);
    }
    // Check out input epsilons in the second graph
    for (auto j : second.out(curr.second)) {
      if (second.ilabel(j) != Graph::epsilon) {
        if (second.ilabelSorted()) {
          // Graph::epsilon < 0
          break;
        } else {
          continue;
        }
      }
      // We only advance along the second arc.
      auto dn1 = curr.first;
      auto dn2 = second.dstNode(j);
      auto idx = toIndex(dn1, dn2, first);
      if (!reachable[idx]) {
        continue;
      }
      if (newNodes[idx] < 0) {
        newNodes[idx] = ngraph.addNode(
            first.isStart(dn1) && second.isStart(dn2),
            first.isAccept(dn1) && second.isAccept(dn2));
        toExplore.emplace(dn1, dn2);
      }
      auto weight = second.weight(j);
      auto newarc = ngraph.addArc(
          currNode, newNodes[idx], Graph::epsilon, second.olabel(j), weight);
      gradInfo.emplace_back(-1, j);
    }
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
