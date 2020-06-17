#include <cstdint>
#include <unordered_set>

#include "autograd.h"

namespace gtn {

void backward(Graph graph) {
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

  // Seed the initial deltas
  auto deltas = Graph::deepCopy(graph);
  deltas.setCalcGrad(false);
  for (auto& a : deltas.arcs()) {
    a.setWeight(1.0);
  }
  graph.addGrad(deltas);

  // Compute gradients
  for (auto iter = tape.rbegin(); iter != tape.rend(); iter++) {
    if (iter->gradFunc()) {
      iter->gradFunc()(iter->inputs(), iter->grad());
    }
  }
}
} // namespace gtn
