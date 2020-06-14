#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <set>

#include "functions.h"

namespace gtn {

Graph clone(Graph other, Projection projection /* = Projection::NONE */) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    if (inputs[0].calcGrad()) {
      for (int i = 0; i < deltas.numArcs(); i++) {
        inputs[0].arcs()[i].addGrad(deltas.arcs()[i].grad());
      }
    }
  };
  Graph g(gradFunc, {other});
  copy(other, projection, g);
  return g;
}

Graph copy(Graph other, Projection projection /* = Projection::NONE */) {
  Graph g;
  copy(other, projection, g);
  return g;
}

void copy(Graph other, Projection projection, Graph& out) {
  for (const auto& node : other.nodes()) {
    out.addNode(node.start(), node.accept());
  }
  for (const auto& arc : other.arcs()) {
    out.addArc(
        arc.upNode()->index(),
        arc.downNode()->index(),
        projection == Projection::OUTPUT ? arc.olabel() : arc.ilabel(),
        projection == Projection::INPUT ? arc.ilabel() : arc.olabel(),
        arc.weight());
  }
}

Graph projectInput(Graph other) {
  return clone(other, Projection::INPUT);
}

Graph projectOutput(Graph other) {
  return clone(other, Projection::OUTPUT);
}

Graph add(Graph lhs, Graph rhs) {
  float weight = lhs.item() + rhs.item();
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    auto delta = deltas.arcs()[0].grad();
    if (inputs[0].calcGrad()) {
      inputs[0].arcs()[0].addGrad(delta);
    }
    if (inputs[1].calcGrad()) {
      inputs[1].arcs()[0].addGrad(delta);
    }
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
    auto delta = deltas.arcs()[0].grad();
    if (inputs[0].calcGrad()) {
      inputs[0].arcs()[0].addGrad(delta);
    }
    if (inputs[1].calcGrad()) {
      inputs[1].arcs()[0].addGrad(-delta);
    }
  };
  Graph result(gradFunc, {lhs, rhs});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, weight);
  return result;
}

Graph closure(Graph graph) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    // Arcs in the new graph were added in the same order.
    for (int i = 0; i < inputs[0].numArcs(); i++) {
      auto grad = deltas.arcs()[i].grad();
      inputs[0].arcs()[i].addGrad(grad);
    }
  };

  Graph closed(gradFunc, {graph});
  closed.addNode(true, true);
  for (auto& n : graph.nodes()) {
    closed.addNode(false, n.accept());
  }
  for (auto& arc : graph.arcs()) {
    closed.addArc(
        arc.upNode()->index() + 1,
        arc.downNode()->index() + 1,
        arc.ilabel(),
        arc.olabel(),
        arc.weight());
  }
  // Add new arcs
  for (auto s : graph.start()) {
    // Epsilon from new start to all old starts
    closed.addArc(0, s->index() + 1, Graph::epsilon);
    for (auto a : graph.accept()) {
      // Epsilon from all accept to all old starts
      closed.addArc(a->index() + 1, s->index() + 1, Graph::epsilon);
    }
  }
  return closed;
}

Graph sum(std::vector<Graph> graphs) {
  auto gradFunc = [](std::vector<Graph>& inputs, Graph& deltas) {
    int gidx = 0;
    for (auto& graph : inputs) {
      if (!graph.calcGrad()) {
        continue;
      }
      for (auto& n : graph.nodes()) {
        // The out arcs of the node will be in the same order
        auto deltan = deltas.node(n.index() + gidx);
        for (int i = 0; i < n.numOut(); i++) {
          auto grad = deltan->out()[i]->grad();
          n.out()[i]->addGrad(grad);
        }
      }
      gidx += graph.numNodes();
    }
  };

  Graph summed(gradFunc, graphs);

  // Add all the nodes in a predictable order
  int gidx = 0;
  for (auto& graph : graphs) {
    for (auto& n : graph.nodes()) {
      summed.addNode(n.start(), n.accept());
    }
    for (auto& arc : graph.arcs()) {
      summed.addArc(
          gidx + arc.upNode()->index(),
          gidx + arc.downNode()->index(),
          arc.ilabel(),
          arc.olabel(),
          arc.weight());
    }
    gidx += graph.numNodes();
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
  auto label_match = [ilabel, olabel](auto a) {
    return a->ilabel() == ilabel && a->olabel() == olabel;
  };

  std::vector<Node*> nodes(other.numNodes(), nullptr);
  Graph graph;
  for (auto& n : other.nodes()) {
    if (n.start() || !std::all_of(n.in().begin(), n.in().end(), label_match)) {
      nodes[n.index()] = graph.addNode(n.start());
    }
  }

  std::queue<Node*> toExplore; // Keep track of where we need to go
  std::set<Node*> reachable; // Keep track of where we've been
  for (auto& n : other.nodes()) {
    auto curr = nodes[n.index()];
    if (curr) {
      toExplore.push(&n);
      reachable.insert(&n);
    }
    while (!toExplore.empty()) {
      auto next = toExplore.front();
      toExplore.pop();
      if (next->accept()) {
        graph.makeAccept(curr);
      }
      for (auto arc : next->out()) {
        if (label_match(arc)) {
          if (!reachable.count(arc->downNode())) {
            toExplore.push(arc->downNode());
            reachable.insert(arc->downNode());
          }
        } else {
          // Add the arc
          graph.addArc(
              curr,
              nodes[arc->downNode()->index()],
              arc->ilabel(),
              arc->olabel());
        }
      }
    }
    reachable.clear();
  }
  return graph;
}

inline size_t toIndex(Node* n1, Node* n2, const Graph& g) {
  return n1->index() + g.numNodes() * n2->index();
}

/* Find any state in the new composed graph which can reach
 * an accepting state. */
auto findReachable(Graph first, Graph second) {
  std::vector<bool> reachable(first.numNodes() * second.numNodes(), false);
  std::queue<std::pair<Node*, Node*>> toExplore;
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
    for (auto a1 : curr.first->in()) {
      for (auto a2 : curr.second->in()) {
        if (a1->olabel() != a2->ilabel()) {
          continue;
        }
        epsilon_matched |= a1->olabel() == Graph::epsilon;
        auto un1 = a1->upNode();
        auto un2 = a2->upNode();
        auto idx = toIndex(un1, un2, first);
        if (!reachable[idx]) {
          // If we haven't seen this state before, explore it.
          toExplore.emplace(un1, un2);
        }
        reachable[idx] = true;
      }
    }
    if (!epsilon_matched) {
      for (auto a1 : curr.first->in()) {
        if (a1->olabel() != Graph::epsilon) {
          continue;
        }
        auto un1 = a1->upNode();
        auto idx = toIndex(un1, curr.second, first);
        if (!reachable[idx]) {
          // If we haven't seen this state before, explore it.
          toExplore.emplace(un1, curr.second);
        }
        reachable[idx] = true;
      }
    }
    if (!epsilon_matched) {
      for (auto a2 : curr.second->in()) {
        if (a2->ilabel() != Graph::epsilon) {
          continue;
        }
        auto un2 = a2->upNode();
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
  std::vector<Node*> newNodes(first.numNodes() * second.numNodes(), nullptr);
  std::queue<std::pair<Node*, Node*>> toExplore;
  for (auto s1 : first.start()) {
    for (auto s2 : second.start()) {
      auto idx = toIndex(s1, s2, first);
      if (reachable[idx]) {
        newNodes[idx] = ngraph.addNode(true, s1->accept() && s2->accept());
        toExplore.emplace(s1, s2);
      }
    }
  }

  std::vector<std::tuple<Arc*, Arc*, Arc*>> gradInfo;
  while (!toExplore.empty()) {
    auto curr = toExplore.front();
    toExplore.pop();
    auto currNode = newNodes[toIndex(curr.first, curr.second, first)];

    for (auto a1 : curr.first->out()) {
      for (auto a2 : curr.second->out()) {
        if (a1->olabel() != a2->ilabel()) {
          continue;
        }
        auto dn1 = a1->downNode();
        auto dn2 = a2->downNode();
        // Ignore if we can't get to an accept state.
        auto idx = toIndex(dn1, dn2, first);
        if (!reachable[idx]) {
          continue;
        }
        // Build the node
        if (newNodes[idx] == nullptr) {
          newNodes[idx] = ngraph.addNode(
              dn1->start() && dn2->start(), dn1->accept() && dn2->accept());
          toExplore.emplace(dn1, dn2);
        }
        auto weight = a1->weight() + a2->weight();
        auto newarc = ngraph.addArc(
            currNode, newNodes[idx], a1->ilabel(), a2->olabel(), weight);
        // Arcs remember where they came from for
        // easy gradient computation.
        gradInfo.emplace_back(a1, a2, newarc);
      }
    }
    // Check for output epsilons in the first graph
    for (auto a1 : curr.first->out()) {
      if (a1->olabel() != Graph::epsilon) {
        continue;
      }
      // We only advance along the first arc.
      auto dn1 = a1->downNode();
      auto dn2 = curr.second;
      auto idx = toIndex(dn1, dn2, first);
      if (!reachable[idx]) {
        continue;
      }
      if (newNodes[idx] == nullptr) {
        newNodes[idx] = ngraph.addNode(
            dn1->start() && dn2->start(), dn1->accept() && dn2->accept());
        toExplore.emplace(dn1, dn2);
      }
      auto weight = a1->weight();
      auto newarc = ngraph.addArc(
          currNode, newNodes[idx], a1->ilabel(), Graph::epsilon, weight);
      gradInfo.emplace_back(a1, nullptr, newarc);
    }
    // Check out input epsilons in the second graph
    for (auto a2 : curr.second->out()) {
      if (a2->ilabel() != Graph::epsilon) {
        continue;
      }
      // We only advance along the second arc.
      auto dn1 = curr.first;
      auto dn2 = a2->downNode();
      auto idx = toIndex(dn1, dn2, first);
      if (!reachable[idx]) {
        continue;
      }
      if (newNodes[idx] == nullptr) {
        newNodes[idx] = ngraph.addNode(
            dn1->start() && dn2->start(), dn1->accept() && dn2->accept());
        toExplore.emplace(dn1, dn2);
      }
      auto weight = a2->weight();
      auto newarc = ngraph.addArc(
          currNode, newNodes[idx], Graph::epsilon, a2->olabel(), weight);
      gradInfo.emplace_back(nullptr, a2, newarc);
    }
  }

  /* Here we assume deltas is the output (e.g. ngraph) and we know where
   * each arc came from. This makes it possible to disambiguate two arcs in the
   * composed graph with the same label and the same src and destination nodes.
   * (TODO we may want to merge these arcs in general, though this may be
   * better implemented in a more explicit way with e.g. minimize.)  */
  auto gradFunc = [gradInfo = std::move(gradInfo)](
      std::vector<Graph> & inputs, Graph /* deltas */) {
    // In this case the arc's parents are always from the
    // first and second input graphs respectively.
    // delta are unused since we have pointers to them in gradInfo
    bool calcGradFirst = inputs[0].calcGrad();
    bool calcGradSecond = inputs[1].calcGrad();
    for (auto& arcs : gradInfo) {
      auto grad = std::get<2>(arcs)->grad();
      if (calcGradFirst && std::get<0>(arcs)) {
        std::get<0>(arcs)->addGrad(grad);
      }
      if (calcGradSecond && std::get<1>(arcs)) {
        std::get<1>(arcs)->addGrad(grad);
      }
    }
  };
  return Graph(ngraph, gradFunc, {first, second});
}

static const float neginf = -std::numeric_limits<float>::infinity();

inline float logadd(float a, float b) {
  if (a == neginf) {
    return b;
  }
  if (b == neginf) {
    return a;
  }
  return std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
}

void forwardGrad(Graph input, Graph output, std::vector<float>& scores) {
  std::queue<Node*> computed;
  auto delta = output.arcs()[0].grad();
  std::vector<int> degrees;
  degrees.reserve(input.numNodes());
  std::vector<float> grads(input.numNodes(), 0.0);
  for (auto& n : input.nodes()) {
    degrees[n.index()] = n.numOut();
    if (n.accept()) {
      grads[n.index()] = delta * std::exp(scores[n.index()] - output.item());
      if (n.numOut() == 0) {
        computed.push(&n);
      }
    }
  }

  while (!computed.empty()) {
    auto node = computed.front();
    computed.pop();
    auto score = scores[node->index()];
    auto grad = grads[node->index()];
    for (auto arc : node->in()) {
      auto un = arc->upNode();
      auto arcGrad =
          grad * std::exp(arc->weight() + scores[un->index()] - score);
      arc->addGrad(arcGrad);
      grads[un->index()] += arcGrad;
      if ((--degrees[un->index()]) == 0) {
        computed.push(un);
      }
    }
  }
}

Graph forward(Graph graph) {
  std::queue<Node*> computed;
  // List of scores and list of in degrees for each node
  std::vector<float> scores(graph.numNodes(), neginf);
  std::vector<int> degrees;
  degrees.reserve(graph.numNodes());
  for (auto& n : graph.nodes()) {
    degrees[n.index()] = n.numIn();
    if (n.start()) {
      scores[n.index()] = 0.0;
      if (n.numIn() == 0) {
        computed.push(&n);
      }
    }
  }

  while (!computed.empty()) {
    auto node = computed.front();
    computed.pop();
    auto score = scores[node->index()];
    for (auto arc : node->out()) {
      auto dn = arc->downNode();
      scores[dn->index()] = logadd(score + arc->weight(), scores[dn->index()]);
      if ((--degrees[dn->index()]) == 0) {
        computed.push(dn);
      }
    }
  }

  // Accumulate scores at all the accept nodes.
  float score = neginf;
  for (auto a : graph.accept()) {
    if (degrees[a->index()] > 0) {
      throw std::invalid_argument(
          "Graph has a cycle, self-loop or is disconnected!");
    }
    score = logadd(score, scores[a->index()]);
  }

  auto gradFunc = [scores = std::move(scores)](
      std::vector<Graph> & inputs, Graph deltas) mutable {
    forwardGrad(inputs[0], deltas, scores);
  };

  Graph result(gradFunc, {graph});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, score);
  return result;
}

} // namespace gtn
