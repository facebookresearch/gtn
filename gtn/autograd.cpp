/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <unordered_set>

#include "gtn/autograd.h"

namespace gtn {

namespace {

void backwardImpl(Graph g, bool retainGraph) {
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

  recurse(g);

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

void backward(Graph g, bool retainGraph /* = false */) {
  // Seed the initial deltas
  auto deltas = std::vector<float>(g.numArcs(), 1.0);
  g.addGrad(std::move(deltas));
  backwardImpl(g, retainGraph);
}

void backward(Graph g, const Graph& grad, bool retainGraph /* = false */) {
  g.addGrad(grad);
  backwardImpl(g, retainGraph);
}

} // namespace gtn
