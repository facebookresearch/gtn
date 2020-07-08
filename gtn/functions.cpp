#include <algorithm>
#include <queue>
#include <set>

#include "compose.h"
#include "functions.h"
#include "shortest.h"

namespace gtn {

Graph negate(const Graph& other) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(negate(deltas));
  };
  Graph result(gradFunc, {other});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, -other.item());
  return result;
}

Graph add(const Graph& lhs, const Graph& rhs) {
  float weight = lhs.item() + rhs.item();
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(deltas);
    inputs[1].addGrad(deltas);
  };
  Graph result(gradFunc, {lhs, rhs});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, weight);
  return result;
}

Graph subtract(const Graph& lhs, const Graph& rhs) {
  float weight = lhs.item() - rhs.item();
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(deltas);
    if (inputs[1].calcGrad()) {
      inputs[1].addGrad(negate(deltas));
    }
  };
  Graph result(gradFunc, {lhs, rhs});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, weight);
  return result;
}

Graph clone(
    const Graph& other,
    Projection projection /* = Projection::NONE */) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(deltas);
  };
  Graph out(gradFunc, {other.withoutWeights()});
  for (auto n = 0; n < other.numNodes(); ++n) {
    out.addNode(other.start(n), other.accept(n));
  }
  for (auto a = 0; a < other.numArcs(); ++a) {
    out.addArc(
        other.upNode(a),
        other.downNode(a),
        projection == Projection::OUTPUT ? other.olabel(a) : other.ilabel(a),
        projection == Projection::INPUT ? other.ilabel(a) : other.olabel(a),
        other.weight(a));
  }
  return out;
}

Graph projectInput(const Graph& other) {
  return clone(other, Projection::INPUT);
}

Graph projectOutput(const Graph& other) {
  return clone(other, Projection::OUTPUT);
}

Graph concat(const Graph& lhs, const Graph& rhs) {
  return concat({lhs, rhs});
}

Graph concat(const std::vector<Graph>& graphs) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    auto grad = deltas.weights();
    for (auto i = 0; i < inputs.size(); ++i) {
      auto& graph = inputs[i];
      if (graph.calcGrad()) {
        graph.addGrad(
          std::vector<float>(grad, grad + graph.numArcs()));
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
          (i == 0) && graph.start(n),
          (i == graphs.size() - 1) && graph.accept(n));
    }
    for (auto a = 0; a < graph.numArcs(); ++a) {
      out.addArc(
          nodeOffset + graph.upNode(a),
          nodeOffset + graph.downNode(a),
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
          out.addArc(a + pNodeOffset, s + nodeOffset, Graph::epsilon);
        }
      }
    }
    nodeOffset += graph.numNodes();
  }
  return out;
}

Graph closure(const Graph& graph) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    auto grad = deltas.weights();
    // *NB* this assumes arcs in the new graph are the same order
    // as in the old graph.
    inputs[0].addGrad(std::vector<float>(grad, grad + inputs[0].numArcs()));
  };

  Graph closed(gradFunc, {graph.withoutWeights()});
  closed.addNode(true, true);
  for (auto n = 0; n < graph.numNodes(); ++n) {
    closed.addNode();
  }
  for (auto a = 0; a < graph.numArcs(); ++a) {
    closed.addArc(
        graph.upNode(a) + 1,
        graph.downNode(a) + 1,
        graph.ilabel(a),
        graph.olabel(a),
        graph.weight(a));
  }

  // Epsilon from new start to old accepts
  for (auto s: graph.start()) {
    closed.addArc(0, s+1, Graph::epsilon);
  }
  // Epsilon from old accepts to new start
  for (auto a : graph.accept()) {
    closed.addArc(a + 1, 0, Graph::epsilon);
  }
  return closed;
}

Graph sum(const std::vector<Graph>& graphs) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    auto grad = deltas.weights();
    for (auto& graph : inputs) {
      if (graph.calcGrad()) {
        graph.addGrad(
          std::vector<float>(grad, grad + graph.numArcs()));
      }
      grad += graph.numArcs();
    }
  };

  std::vector<Graph> inputs;
  for (auto& g : graphs) {
    inputs.push_back(g.withoutWeights());
  }
  Graph summed(gradFunc, std::move(inputs));

  // Add all the nodes in a predictable order
  int nodeOffset = 0;
  for (auto& graph : graphs) {
    for (auto n = 0; n < graph.numNodes(); ++n) {
      summed.addNode(graph.start(n), graph.accept(n));
    }
    for (auto a = 0; a < graph.numArcs(); ++a) {
      summed.addArc(
          nodeOffset + graph.upNode(a),
          nodeOffset + graph.downNode(a),
          graph.ilabel(a),
          graph.olabel(a),
          graph.weight(a));
    }
    nodeOffset += graph.numNodes();
  }

  return summed;
}

Graph compose(const Graph& lhs, const Graph& rhs) {
  std::shared_ptr<detail::ArcMatcher> matcher;
  bool lhsSorted = lhs.olabelSorted();
  bool rhsSorted = rhs.ilabelSorted();
  if (lhsSorted && rhsSorted) {
    matcher = std::make_shared<detail::DoublySortedMatcher>(lhs, rhs);
  } else if (lhsSorted || rhsSorted) {
    matcher = std::make_shared<detail::SinglySortedMatcher>(lhs, rhs, lhsSorted);
  } else {
    matcher = std::make_shared<detail::UnsortedMatcher>(lhs, rhs);
  }
  return detail::compose(lhs, rhs, matcher);
}

Graph intersect(const Graph& lhs, const Graph& rhs) {
  std::shared_ptr<detail::ArcMatcher> matcher;
  bool lhsSorted = lhs.ilabelSorted() || lhs.olabelSorted();
  bool rhsSorted = rhs.ilabelSorted() || rhs.olabelSorted();
  if (lhsSorted && rhsSorted) {
    matcher = std::make_shared<detail::DoublySortedMatcher>(lhs, rhs);
  } else if (lhsSorted || rhsSorted) {
    matcher = std::make_shared<detail::SinglySortedMatcher>(lhs, rhs, lhsSorted);
  } else {
    matcher = std::make_shared<detail::UnsortedMatcher>(lhs, rhs);
  }
  return detail::compose(lhs, rhs, matcher);
}

Graph remove(const Graph& other, int label /* = Graph::epsilon */) {
  return remove(other, label, label);
}

Graph remove(const Graph& other, int ilabel, int olabel) {
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
  auto label_match = [&other, ilabel, olabel](auto a) {
    return other.ilabel(a) == ilabel && other.olabel(a) == olabel;
  };

  std::vector<int> nodes(other.numNodes(), -1);
  Graph graph;
  for (auto n = 0; n < other.numNodes(); ++n) {
    if (other.start(n) ||
        !std::all_of(other.in(n).begin(), other.in(n).end(), label_match)) {
      nodes[n] =
          graph.addNode(other.start(n));
    }
  }

  std::queue<int> toExplore; // Keep track of where we need to go
  std::set<int> reachable; // Keep track of where we've been
  for (auto n = 0; n < other.numNodes(); ++n) {
    auto curr = nodes[n];
    if (curr >= 0) {
      toExplore.push(n);
      reachable.insert(n);
    }
    while (!toExplore.empty()) {
      auto next = toExplore.front();
      toExplore.pop();
      if (other.accept(next)) {
        graph.makeAccept(curr);
      }
      for (auto a : other.out(next)) {
        auto dn = other.downNode(a);
        if (label_match(a)) {
          if (!reachable.count(dn)) {
            toExplore.push(dn);
            reachable.insert(dn);
          }
        } else {
          // Add the arc
          graph.addArc(curr, nodes[dn], other.ilabel(a), other.olabel(a));
        }
      }
    }
    reachable.clear();
  }
  return graph;
}

Graph forwardScore(const Graph& graph) {
  return detail::shortestDistance(graph);
}

Graph viterbiScore(const Graph& graph) {
  return detail::shortestDistance(graph, true);
}

Graph viterbiPath(const Graph& graph) {
  return detail::shortestPath(graph);
}

} // namespace gtn
