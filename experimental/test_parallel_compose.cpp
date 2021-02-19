#include <cstdlib>
#include <sstream>

#include "gtn/gtn.h"
#include "parallel_compose.h"

using namespace gtn;

// *NB* num_arcs is assumed to be greater than num_nodes.
Graph makeRandomDAG(int num_nodes, int num_arcs) {
  Graph graph;
  graph.addNode(true);
  for (int n = 1; n < num_nodes; n++) {
    graph.addNode(false, n == num_nodes - 1);
    graph.addArc(n - 1, n, 0); // assure graph is connected
  }
  for (int i = 0; i < num_arcs - num_nodes + 1; i++) {
    // To preserve DAG property, select src then select dst to be
    // greater than source.
    // select from [0, num_nodes-2]:
    auto src = rand() % (num_nodes - 1);
    // then select from  [src + 1, num_nodes - 1]:
    auto dst = src + 1 + rand() % (num_nodes - src - 1);
    graph.addArc(src, dst, 0);
  }
  return graph;
}

void testConversion() {
  using gtn::detail::dataparallel::convertToDataParallel;
  using gtn::detail::dataparallel::convertFromDataParallel;

  auto check = [](const Graph& gIn) {
    auto gOut = convertFromDataParallel(convertToDataParallel(gIn));
    assert(equal(gIn, gOut));
  };

  {
    // Basic linear graph
    auto gIn = linearGraph(4, 3);
    check(gIn);
  }

  {
    // Empty graph
    Graph gIn;
    check(gIn);

    // Singleton
    gIn.addNode();
    check(gIn);
  }

  {
    // Singleton start node
    Graph gIn;
    gIn.addNode(true);
    check(gIn);
  }

  {
    // Singleton accept node
    Graph gIn;
    gIn.addNode(false, true);
    check(gIn);
  }

  {
    // Singleton start and accept node
    Graph gIn;
    gIn.addNode(true, true);
    check(gIn);
  }

  {
    // Multiple start and accept nodes
    Graph gIn;
    gIn.addNode(true);
    gIn.addNode(false, true);
    gIn.addNode(true);
    gIn.addNode(false, true);
    gIn.addArc(0, 1, 0, 2, 2.0);
    gIn.addArc(0, 1, 0, 2, 3.0);
    gIn.addArc(2, 2, 1, 0, 0.0);
    gIn.addArc(2, 3, 1, 0, 4.0);
    gIn.addArc(1, 3, 0, 0, 6.0);
  }

  {
    // Large random graph
    auto gIn = makeRandomDAG(100, 200);
    check(gIn);
  }
}

void testNoEpsilon() {
  auto check = [](const Graph& g1, const Graph& g2) {
    auto gOut = compose(g1, g2);
    auto gOutP = gtn::detail::dataparallel::compose(g1, g2);
    assert(equal(gOut, gOutP));
  };

  // Check a simple chain graphs
  check(linearGraph(10, 1), linearGraph(10, 1));
  check(linearGraph(10, 2), linearGraph(10, 1));
  check(linearGraph(10, 20), linearGraph(10, 1));

  // Currently fails!
  check(linearGraph(1, 2), linearGraph(1, 2));
}

int main() {
  testConversion();
  std::cout << "Conversion checks passed!" << std::endl;

  testNoEpsilon();
  std::cout << "No epsilon compositions passed!" << std::endl;
}
