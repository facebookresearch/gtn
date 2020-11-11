/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtn/rand.h"
#include "gtn/functions.h"
#include "gtn/utils.h"

namespace gtn {

Graph sample(const Graph& g, size_t maxLength /* = 1000 */) {
  if (!g.numStart() || !g.numAccept()) {
    return Graph{};
  }

  std::vector<int> arcs;
  size_t node = g.start()[rand() % g.numStart()];
  size_t acceptLength = 0;
  for (size_t length = 0; length < maxLength + 1; length++) {
    auto mod = g.numOut(node) + g.isAccept(node);
    acceptLength = g.isAccept(node) ? (length + 1) : acceptLength;

    // Dead end
    if (!mod) {
      return Graph{};
    }

    // Select a random arc with optional transition to "final state" if node is
    // accepting
    auto i = static_cast<int>(rand() % mod);

    // Successful and complete
    if (i == g.numOut(node)) {
      break;
    }

    auto arc = g.out(node, i);
    node = g.dstNode(arc);
    arcs.push_back(arc);
  }

  // No accepting path
  if (!acceptLength) {
    return Graph{};
  }

  arcs.resize(acceptLength - 1);
  auto gradFunc = [arcs = arcs](std::vector<Graph>& inputs, Graph deltas) {
    if (inputs[0].calcGrad()) {
      auto grad = std::vector<float>(inputs[0].numArcs(), 0.0);
      for (auto a = 0; a < deltas.numArcs(); ++a) {
        // The arcs in deltas should are the same order as in arcs
        grad[arcs[a]] += deltas.weight(a);
      }
      inputs[0].addGrad(std::move(grad));
    }
  };

  // Build the graph
  Graph path(gradFunc, {g});
  path.addNode(true, acceptLength == 1);
  for (int i = 1; i < acceptLength; i++) {
    path.addNode(false, (i + 1) == acceptLength);
    path.addArc(
        i - 1,
        i,
        g.ilabel(arcs[i - 1]),
        g.olabel(arcs[i - 1]),
        g.weight(arcs[i - 1]));
  }
  return path;
}

bool randEquivalent(
    const Graph& g1,
    const Graph& g2,
    size_t numSamples /* = 100 */,
    double tol /* = 1e-4 */,
    size_t maxLength /* = 1000 */) {
  for (int i = 0; i < numSamples; i++) {
    auto path = sample(rand() % 2 ? g1 : g2, maxLength);
    path.setCalcGrad(false);

    // Ignore empty paths
    if (equal(path, Graph{})) {
      continue;
    }

    auto inp = projectInput(path);
    auto outp = projectOutput(path);

    auto composedG1 = compose(inp, g1);
    composedG1.setCalcGrad(false);
    composedG1 = compose(composedG1, outp);

    auto composedG2 = compose(inp, g2);
    composedG2.setCalcGrad(false);
    composedG2 = compose(composedG2, outp);

    auto isEmptyG1 = equal(composedG1, Graph{});
    auto isEmptyG2 = equal(composedG2, Graph{});

    // Only one of the graphs is empty
    if (isEmptyG1 != isEmptyG2) {
      return false;
    }

    // Both graphs are empty
    if (isEmptyG1 && isEmptyG2) {
      continue;
    }

    auto scoreG1 = forwardScore(composedG1).item();
    auto scoreG2 = forwardScore(composedG2).item();

    // Check within tolerance
    if (abs(scoreG1 - scoreG2) > tol) {
      return false;
    }
  }

  return true;
}

} // namespace gtn
