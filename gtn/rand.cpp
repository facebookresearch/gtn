#include "gtn/rand.h"
#include "gtn/functions.h"
#include "gtn/utils.h"

namespace gtn {

Graph sample(const Graph& graph, size_t maxLength /* = 1000 */) {
  if (!graph.numStart() || !graph.numAccept()) {
    return Graph{};
  }

  std::vector<int> arcs;
  auto node = graph.start()[rand() % graph.numStart()];
  size_t acceptLength = 0;
  for (size_t length = 0; length < maxLength + 1; length++) {
    auto mod = graph.numOut(node) + graph.accept(node);
    acceptLength = graph.accept(node) ? (length + 1) : acceptLength;

    // Dead end
    if (!mod) {
      return Graph{};
    }

    // Select a random arc with optional transition to "final state" if node is
    // accepting
    auto i = rand() % mod;

    // Successful and complete
    if (i == graph.numOut(node)) {
      break;
    }

    auto arc = graph.out(node, i);
    node = graph.dstNode(arc);
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
  Graph path(gradFunc, {graph});
  path.addNode(true, acceptLength == 1);
  for (int i = 1; i < acceptLength; i++) {
    path.addNode(false, (i + 1) == acceptLength);
    path.addArc(
        i - 1,
        i,
        graph.ilabel(arcs[i - 1]),
        graph.olabel(arcs[i - 1]),
        graph.weight(arcs[i - 1]));
  }
  return path;
}

bool randEquivalent(
    const Graph& a,
    const Graph& b,
    size_t numSamples /* = 100 */,
    double tol /* = 1e-4 */,
    size_t maxLength /* = 1000 */) {
  for (int i = 0; i < numSamples; i++) {
    auto path = sample(rand() % 2 ? a : b, maxLength);
    path.setCalcGrad(false);

    // Ignore empty paths
    if (equal(path, Graph{})) {
      continue;
    }

    auto inp = projectInput(path);
    auto outp = projectOutput(path);

    auto composedA = compose(inp, a);
    composedA.setCalcGrad(false);
    composedA = compose(composedA, outp);

    auto composedB = compose(inp, b);
    composedB.setCalcGrad(false);
    composedB = compose(composedB, outp);

    auto isEmptyA = equal(composedA, Graph{});
    auto isEmptyB = equal(composedB, Graph{});

    // Only one of the graphs is empty
    if (isEmptyA != isEmptyB) {
      return false;
    }

    // Both graphs are empty
    if (isEmptyA && isEmptyB) {
      continue;
    }

    auto scoreA = forwardScore(composedA).item();
    auto scoreB = forwardScore(composedB).item();

    // Check within tolerance
    if (abs(scoreA - scoreB) > tol) {
      return false;
    }
  }

  return true;
}

} // namespace gtn
