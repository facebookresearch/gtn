#include <cstdint>
#include <unordered_set>

#include "autograd.h"

namespace gtn {

namespace {

void backwardImpl(Graph graph, bool retainGraph) {
  // Build the tape
  std::unordered_set<std::uintptr_t> cache;
  std::vector<Graph> tape;

  std::function<void(Graph&)> recurse;

  // Topological sort
  recurse = [&](Graph& g) {
    auto id = g.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    for (auto& input : g.inputs()) {
      recurse(input);
    }
    cache.insert(id);
    tape.push_back(g);
  };

  recurse(graph);

  // Compute gradients
  for (auto iter = tape.rbegin(); iter != tape.rend(); iter++) {
    if (iter->gradFunc()) {
      if (iter->inputs().empty()) {
        throw std::invalid_argument(
            "[autograd::backward] Cannot Backward twice without retaining the graph.");
      }
      iter->gradFunc()(iter->inputs(), iter->grad());
      if (!retainGraph) {
        iter->inputs().clear();
        *iter = Graph{};
      }
    }
  }
}

} // namespace

void backward(Graph graph, bool retainGraph /* = false */) {
  // Seed the initial deltas
  auto deltas = std::vector<float>(graph.numArcs(), 1.0);
  graph.addGrad(std::move(deltas));
  backwardImpl(graph, retainGraph);
}

void backward(Graph graph, const Graph& grad, bool retainGraph /* = false */) {
  graph.addGrad(grad);
  backwardImpl(graph, retainGraph);
}

} // namespace gtn
