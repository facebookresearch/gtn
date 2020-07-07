#include <algorithm>
#include <queue>
#include <set>

#include "functions.h"
#include "shortest.h"

namespace gtn {

Graph negate(Graph other) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    inputs[0].addGrad(negate(deltas));
  };
  Graph result(gradFunc, {other});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, -other.item());
  return result;
}

Graph add(Graph lhs, Graph rhs) {
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

Graph subtract(Graph lhs, Graph rhs) {
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

Graph clone(Graph other, Projection projection /* = Projection::NONE */) {
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

Graph projectInput(Graph other) {
  return clone(other, Projection::INPUT);
}

Graph projectOutput(Graph other) {
  return clone(other, Projection::OUTPUT);
}

Graph concat(Graph lhs, Graph rhs) {
  return concat({lhs, rhs});
}

Graph concat(std::vector<Graph> graphs) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    int arcOffset = 0;
    for (auto i = 0; i < inputs.size(); ++i) {
      auto& graph = inputs[i];
      if (graph.calcGrad()) {
        auto grad = std::vector<float>(graph.numArcs());
        for (auto a = 0; a < grad.size(); ++a) {
          grad[a] = deltas.weight(a + arcOffset);
        }
        graph.addGrad(std::move(grad));
      }
      arcOffset += graph.numArcs();
      if (i > 0) {
        arcOffset += inputs[i - 1].numAccept() * graph.numStart();
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

Graph closure(Graph graph) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    auto grad = std::vector<float>(inputs[0].numArcs());
    // *NB* this assumes arcs in the new graph are the same order
    // as in the old graph.
    for (auto i = 0; i < grad.size(); ++i) {
      grad[i] = deltas.weight(i);
    }
    inputs[0].addGrad(std::move(grad));
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

Graph sum(std::vector<Graph> graphs) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    int arcOffset = 0;
    for (auto& graph : inputs) {
      if (graph.calcGrad()) {
        auto grad = std::vector<float>(graph.numArcs());
        for (auto a = 0; a < grad.size(); ++a) {
          grad[a] = deltas.weight(a + arcOffset);
        }
        graph.addGrad(std::move(grad));
      }
      arcOffset += graph.numArcs();
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

Graph remove(Graph other, int label /* = Graph::epsilon */) {
  return remove(other, label, label);
}

Graph remove(Graph other, int ilabel, int olabel) {
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

inline size_t toIndex(int n1, int n2, const Graph& g) {
  return n1 + g.numNodes() * n2;
}

/* Find any state in the new composed graph which can reach
 * an accepting state. */
auto findReachable(Graph first, Graph second) {
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
    for (auto i : first.in(curr.first)) {
      for (auto j : second.in(curr.second)) {
        if (first.olabel(i) != second.ilabel(j)) {
          continue;
        }
        epsilon_matched |= first.olabel(i) == Graph::epsilon;
        auto un1 = first.upNode(i);
        auto un2 = second.upNode(j);
        auto idx = toIndex(un1, un2, first);
        if (!reachable[idx]) {
          // If we haven't seen this state before, explore it.
          toExplore.emplace(un1, un2);
        }
        reachable[idx] = true;
      }
    }
    if (!epsilon_matched) {
      for (auto i : first.in(curr.first)) {
        if (first.olabel(i) != Graph::epsilon) {
          continue;
        }
        auto un1 = first.upNode(i);
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
          continue;
        }
        auto un2 = second.upNode(j);
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

// Composes two graphs and returns a new graph
Graph compose(Graph first, Graph second) {
  // Compute reachable nodes from any accept state in the new graph
  auto reachable = findReachable(first, second);

  // Compose the graphs
  Graph ngraph;
  std::vector<int> newNodes(first.numNodes() * second.numNodes(), -1);
  std::queue<std::pair<int, int>> toExplore;
  for (auto s1 : first.start()) {
    for (auto s2 : second.start()) {
      auto idx = toIndex(s1, s2, first);
      if (reachable[idx]) {
        newNodes[idx] =
            ngraph.addNode(true, first.accept(s1) && second.accept(s2));
        toExplore.emplace(s1, s2);
      }
    }
  }
  std::vector<std::pair<int, int>> gradInfo;
  while (!toExplore.empty()) {
    auto curr = toExplore.front();
    toExplore.pop();
    auto currNode = newNodes[toIndex(curr.first, curr.second, first)];

    for (auto i : first.out(curr.first)) {
      for (auto j : second.out(curr.second)) {
        if (first.olabel(i) != second.ilabel(j)) {
          continue;
        }
        auto dn1 = first.downNode(i);
        auto dn2 = second.downNode(j);
        // Ignore if we can't get to an accept state.
        auto idx = toIndex(dn1, dn2, first);
        if (!reachable[idx]) {
          continue;
        }
        // Build the node
        if (newNodes[idx] < 0) {
          newNodes[idx] = ngraph.addNode(
              first.start(dn1) && second.start(dn2),
              first.accept(dn1) && second.accept(dn2));
          toExplore.emplace(dn1, dn2);
        }
        auto weight = first.weight(i) + second.weight(j);
        auto newarc = ngraph.addArc(
            currNode, newNodes[idx], first.ilabel(i), second.olabel(j), weight);
        // Arcs remember where they came from for
        // easy gradient computation.
        gradInfo.emplace_back(i, j);
      }
    }
    // Check for output epsilons in the first graph
    for (auto i : first.out(curr.first)) {
      if (first.olabel(i) != Graph::epsilon) {
        continue;
      }
      // We only advance along the first arc.
      auto dn1 = first.downNode(i);
      auto dn2 = curr.second;
      auto idx = toIndex(dn1, dn2, first);
      if (!reachable[idx]) {
        continue;
      }
      if (newNodes[idx] < 0) {
        newNodes[idx] = ngraph.addNode(
            first.start(dn1) && second.start(dn2),
            first.accept(dn1) && second.accept(dn2));
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
        continue;
      }
      // We only advance along the second arc.
      auto dn1 = curr.first;
      auto dn2 = second.downNode(j);
      auto idx = toIndex(dn1, dn2, first);
      if (!reachable[idx]) {
        continue;
      }
      if (newNodes[idx] < 0) {
        newNodes[idx] = ngraph.addNode(
            first.start(dn1) && second.start(dn2),
            first.accept(dn1) && second.accept(dn2));
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
   * (TODO we may want to merge these arcs in general, though this may be
   * better implemented in a more explicit way with e.g. minimize.)  */
  auto gradFunc = [gradInfo = std::move(gradInfo)](
                      std::vector<Graph>& inputs, Graph deltas) {
    // In this case the arc's parents are always from the
    // first and second input graphs respectively.
    bool calcGrad1 = inputs[0].calcGrad();
    bool calcGrad2 = inputs[1].calcGrad();
    if (!(calcGrad1 || calcGrad2)) {
      return;
    }
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
  return Graph(
      ngraph, gradFunc, {first.withoutWeights(), second.withoutWeights()});
}

Graph forwardScore(Graph graph) {
  return detail::shortestDistance(graph);
}

Graph viterbiScore(Graph graph) {
  return detail::shortestDistance(graph, true);
}

Graph viterbiPath(Graph graph) {
  return detail::shortestPath(graph);
}

} // namespace gtn
