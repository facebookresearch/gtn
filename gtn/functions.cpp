#include <algorithm>
#include <queue>
#include <set>

#include "gtn/functions.h"
#include "gtn/functions/compose.h"
#include "gtn/functions/shortest.h"

namespace gtn {

Graph negate(const Graph& g) {
  if (g.numArcs() != 1) {
    throw std::logic_error("[gtn::negate] input must have only one arc");
  }
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(negate(deltas));
  };
  Graph result(gradFunc, {g});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, -g.item());
  return result;
}

Graph add(const Graph& g1, const Graph& g2) {
  if (g1.numArcs() != 1 || g2.numArcs() != 1) {
    throw std::logic_error("[gtn::add] inputs must have only one arc");
  }
  float weight = g1.item() + g2.item();
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(deltas);
    inputs[1].addGrad(deltas);
  };
  Graph result(gradFunc, {g1, g2});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, weight);
  return result;
}

Graph subtract(const Graph& g1, const Graph& g2) {
  if (g1.numArcs() != 1 || g2.numArcs() != 1) {
    throw std::logic_error("[gtn::subtract] inputs must have only one arc");
  }
  float weight = g1.item() - g2.item();
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(deltas);
    if (inputs[1].calcGrad()) {
      inputs[1].addGrad(negate(deltas));
    }
  };
  Graph result(gradFunc, {g1, g2});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, weight);
  return result;
}

Graph clone(const Graph& g, Projection projection /* = Projection::NONE */) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(deltas);
  };
  Graph out(gradFunc, {g.withoutWeights()});
  for (auto n = 0; n < g.numNodes(); ++n) {
    out.addNode(g.isStart(n), g.isAccept(n));
  }
  for (auto a = 0; a < g.numArcs(); ++a) {
    out.addArc(
        g.srcNode(a),
        g.dstNode(a),
        projection == Projection::OUTPUT ? g.olabel(a) : g.ilabel(a),
        projection == Projection::INPUT ? g.ilabel(a) : g.olabel(a),
        g.weight(a));
  }
  return out;
}

Graph projectInput(const Graph& g) {
  return clone(g, Projection::INPUT);
}

Graph projectOutput(const Graph& g) {
  return clone(g, Projection::OUTPUT);
}

Graph concat(const Graph& g1, const Graph& g2) {
  return concat({g1, g2});
}

Graph concat(const std::vector<Graph>& graphs) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    auto grad = deltas.weights();
    for (auto i = 0; i < inputs.size(); ++i) {
      auto& graph = inputs[i];
      if (graph.calcGrad()) {
        graph.addGrad(std::vector<float>(grad, grad + graph.numArcs()));
      }
      grad += graph.numArcs();
      if (i > 0) {
        grad += inputs[i - 1].numAccept() * graph.numStart();
      }
    }
  };

  std::vector<Graph> inputs;
  for (auto& g : graphs) {
    inputs.push_back(g.withoutWeights());
  }
  Graph out(gradFunc, std::move(inputs));

  // By definition a^0 accepts the empty string (epsilon)
  if (graphs.size() == 0) {
    out.addNode(true, true);
    return out;
  }
  int nodeOffset = 0;
  for (auto i = 0; i < graphs.size(); ++i) {
    auto& graph = graphs[i];
    for (auto n = 0; n < graph.numNodes(); ++n) {
      out.addNode(
          (i == 0) && graph.isStart(n),
          (i == graphs.size() - 1) && graph.isAccept(n));
    }
    for (auto a = 0; a < graph.numArcs(); ++a) {
      out.addArc(
          nodeOffset + graph.srcNode(a),
          nodeOffset + graph.dstNode(a),
          graph.ilabel(a),
          graph.olabel(a),
          graph.weight(a));
    }
    // If i > 0 connect graph[i - 1]'s accept states to this graph's
    // starts states
    if (i > 0) {
      auto& pGraph = graphs[i - 1];
      auto pNodeOffset = nodeOffset - pGraph.numNodes();
      for (auto a : pGraph.accept()) {
        for (auto s : graph.start()) {
          out.addArc(a + pNodeOffset, s + nodeOffset, epsilon);
        }
      }
    }
    nodeOffset += graph.numNodes();
  }
  return out;
}

Graph closure(const Graph& g) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    auto grad = deltas.weights();
    // *NB* this assumes arcs in the new graph are the same order
    // as in the old graph.
    inputs[0].addGrad(std::vector<float>(grad, grad + inputs[0].numArcs()));
  };

  Graph closed(gradFunc, {g.withoutWeights()});
  closed.addNode(true, true);
  for (auto n = 0; n < g.numNodes(); ++n) {
    closed.addNode();
  }
  for (auto a = 0; a < g.numArcs(); ++a) {
    closed.addArc(
        g.srcNode(a) + 1,
        g.dstNode(a) + 1,
        g.ilabel(a),
        g.olabel(a),
        g.weight(a));
  }

  // Epsilon from new start to old accepts
  for (auto s : g.start()) {
    closed.addArc(0, s + 1, epsilon);
  }
  // Epsilon from old accepts to new start
  for (auto a : g.accept()) {
    closed.addArc(a + 1, 0, epsilon);
  }
  return closed;
}

Graph union_(const std::vector<Graph>& graphs) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    auto grad = deltas.weights();
    for (auto& graph : inputs) {
      if (graph.calcGrad()) {
        graph.addGrad(std::vector<float>(grad, grad + graph.numArcs()));
      }
      grad += graph.numArcs();
    }
  };

  std::vector<Graph> inputs;
  for (auto& g : graphs) {
    inputs.push_back(g.withoutWeights());
  }
  Graph out(gradFunc, std::move(inputs));

  // Add all the nodes in a predictable order
  int nodeOffset = 0;
  for (auto& graph : graphs) {
    for (auto n = 0; n < graph.numNodes(); ++n) {
      out.addNode(graph.isStart(n), graph.isAccept(n));
    }
    for (auto a = 0; a < graph.numArcs(); ++a) {
      out.addArc(
          nodeOffset + graph.srcNode(a),
          nodeOffset + graph.dstNode(a),
          graph.ilabel(a),
          graph.olabel(a),
          graph.weight(a));
    }
    nodeOffset += graph.numNodes();
  }

  return out;
}

Graph compose(const Graph& g1, const Graph& g2) {
  std::shared_ptr<detail::ArcMatcher> matcher;
  bool g1Sorted = g1.olabelSorted();
  bool g2Sorted = g2.ilabelSorted();
  if (g1Sorted && g2Sorted) {
    matcher = std::make_shared<detail::DoublySortedMatcher>(g1, g2);
  } else if (g1Sorted || g2Sorted) {
    matcher = std::make_shared<detail::SinglySortedMatcher>(g1, g2, g1Sorted);
  } else {
    matcher = std::make_shared<detail::UnsortedMatcher>(g1, g2);
  }
  return detail::compose(g1, g2, matcher);
}

Graph intersect(const Graph& g1, const Graph& g2) {
  std::shared_ptr<detail::ArcMatcher> matcher;
  bool g1Sorted = g1.ilabelSorted() || g1.olabelSorted();
  bool g2Sorted = g2.ilabelSorted() || g2.olabelSorted();
  if (g1Sorted && g2Sorted) {
    matcher = std::make_shared<detail::DoublySortedMatcher>(g1, g2);
  } else if (g1Sorted || g2Sorted) {
    matcher = std::make_shared<detail::SinglySortedMatcher>(g1, g2, g1Sorted);
  } else {
    matcher = std::make_shared<detail::UnsortedMatcher>(g1, g2);
  }
  return detail::compose(g1, g2, matcher);
}

Graph remove(const Graph& g, int label /* = epsilon */) {
  return remove(g, label, label);
}

Graph remove(const Graph& g, int ilabel, int olabel) {
  /* TODO we may want to make this function work appropriately with weights.
   * In order to do so for DAGs, we can modify the routine to accumulate scores
   * of epsilon transitions appropriately. Every time we add a node to the
   * reachable, we logadd the score of the arc + the up node's score into that
   * reachable nodes current score. Then when we explore a node we extract its
   * current score. The current score should be added to all outgoing arc
   * weights.
   * Some complexities arise from:
   * a) do we handle cycles here?
   * b) is there a faster algorithm (all-pairs shortest path) for computing the
   * scores?
   * c) gradient computation may be more complex
   */
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    throw std::logic_error("[gtn::remove] gradient compuation not implemented");
  };

  auto label_match = [&g, ilabel, olabel](auto a) {
    return g.ilabel(a) == ilabel && g.olabel(a) == olabel;
  };

  std::vector<int> nodes(g.numNodes(), -1);
  Graph graph(gradFunc, {g});
  for (auto n = 0; n < g.numNodes(); ++n) {
    if (g.isStart(n) ||
        !std::all_of(g.in(n).begin(), g.in(n).end(), label_match)) {
      nodes[n] = graph.addNode(g.isStart(n));
    }
  }

  std::queue<int> toExplore; // Keep track of where we need to go
  std::set<int> reachable; // Keep track of where we've been
  for (auto n = 0; n < g.numNodes(); ++n) {
    auto curr = nodes[n];
    if (curr >= 0) {
      toExplore.push(n);
      reachable.insert(n);
    }
    while (!toExplore.empty()) {
      auto next = toExplore.front();
      toExplore.pop();
      if (g.isAccept(next)) {
        graph.makeAccept(curr);
      }
      for (auto a : g.out(next)) {
        auto dn = g.dstNode(a);
        if (label_match(a)) {
          if (!reachable.count(dn)) {
            toExplore.push(dn);
            reachable.insert(dn);
          }
        } else {
          // Add the arc
          graph.addArc(curr, nodes[dn], g.ilabel(a), g.olabel(a));
        }
      }
    }
    reachable.clear();
  }
  return graph;
}

Graph forwardScore(const Graph& g) {
  return detail::shortestDistance(g);
}

Graph viterbiScore(const Graph& g) {
  return detail::shortestDistance(g, true);
}

Graph viterbiPath(const Graph& g) {
  return detail::shortestPath(g);
}

} // namespace gtn
