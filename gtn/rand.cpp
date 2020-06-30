#include "rand.h"
#include "functions.h"
#include "utils.h"

namespace gtn {

Graph sample(Graph graph, size_t maxLength /* = 1000 */) {
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
    node = graph.downNode(arc);
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
    Graph a,
    Graph b,
    size_t numSamples /* = 100 */,
    double tol /* = 1e-4 */,
    size_t maxLength /* = 1000 */) {
  // TODO, awni, find a way to break the autograd tape here.. we need to break
  // it on a and b but without clearing their gradFuncs and inputs and ideally
  // without making a deep copy of the graphs.
  for (int i = 0; i < numSamples; i++) {
    auto path = sample(rand() % 2 ? a : b, maxLength);

    // Ignore empty paths
    if (equals(path, Graph{})) {
      continue;
    }

    auto inp = projectInput(path);
    auto outp = projectOutput(path);

    auto composedA = compose(compose(inp, a), outp);
    auto composedB = compose(compose(inp, b), outp);

    auto isEmptyA = equals(composedA, Graph{});
    auto isEmptyB = equals(composedB, Graph{});

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
