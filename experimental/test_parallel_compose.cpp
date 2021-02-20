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
    assert(isomorphic(gOut, gOutP));
  };

  // Empty result
  check(linearGraph(1, 1), linearGraph(2, 1));

  // Accepts empty string
  {
    auto g1 = Graph();
    g1.addNode(true, true);
    auto g2 = Graph();
    g2.addNode(true, true);
    check(g1, g2);
  }

  // Check some simple chain graphs
  check(linearGraph(1, 1), linearGraph(1, 1));
  check(linearGraph(5, 1), linearGraph(5, 1));
  check(linearGraph(5, 2), linearGraph(5, 1));
  check(linearGraph(5, 10), linearGraph(5, 1));
  check(linearGraph(1, 2), linearGraph(1, 2));
  check(linearGraph(5, 2), linearGraph(5, 2));
  check(linearGraph(5, 5), linearGraph(5, 3));
  check(linearGraph(5, 3), linearGraph(5, 5));

  // Check some graphs with self-loops!
  {
    auto g1 = linearGraph(1, 1);
    auto g2 = linearGraph(1, 1);
    g1.addArc(0, 0, 0, 0);
    g1.addArc(1, 1, 0, 0);
    check(g1, g2);

    g2.addArc(0, 0, 0, 0);
    g2.addArc(1, 1, 0, 0);
    check(g1, g2);
  }

  // More complex test cases
  {
    // Self-loop in the composed graph
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 0, 0);
    g1.addArc(0, 1, 1);
    g1.addArc(1, 1, 2);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0);
    g2.addArc(1, 1, 0);
    g2.addArc(1, 2, 1);

    std::stringstream in(
        "0\n"
        "2\n"
        "0 1 0\n"
        "1 1 0\n"
        "1 2 1\n");
    Graph expected = loadTxt(in);
    assert(isomorphic(
      gtn::detail::dataparallel::compose(g1, g2), expected));
  }

  {
    // Loop in the composed graph
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0);
    g1.addArc(1, 1, 1);
    g1.addArc(1, 0, 0);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 0, 0);
    g2.addArc(0, 1, 1);
    g2.addArc(1, 0, 1);

    std::stringstream in(
        "0\n"
        "2\n"
        "0 1 0\n"
        "1 0 0\n"
        "1 2 1\n"
        "2 1 1\n");
    Graph expected = loadTxt(in);
    assert(isomorphic(
      gtn::detail::dataparallel::compose(g1, g2), expected));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode();
    g1.addNode();
    g1.addNode(false, true);
    for (int i = 0; i < g1.numNodes() - 1; i++) {
      for (int j = 0; j < 3; j++) {
        g1.addArc(i, i + 1, j, j, static_cast<float>(j));
      }
    }

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, 3.5);
    g2.addArc(1, 1, 0, 0, 2.5);
    g2.addArc(1, 2, 1, 1, 1.5);
    g2.addArc(2, 2, 1, 1, 4.5);
    std::stringstream in(
        "0\n"
        "6\n"
        "0 1 0 0 3.5\n"
        "1 2 0 0 2.5\n"
        "1 4 1 1 2.5\n"
        "2 3 0 0 2.5\n"
        "2 5 1 1 2.5\n"
        "4 5 1 1 5.5\n"
        "3 6 1 1 2.5\n"
        "5 6 1 1 5.5\n");
    Graph expected = loadTxt(in);
    // THIS FAILS!
    assert(isomorphic(
      gtn::detail::dataparallel::compose(g1, g2), expected));
  }
}

int main() {
  testConversion();
  std::cout << "Conversion checks passed!" << std::endl;

  testNoEpsilon();
  std::cout << "No epsilon compositions passed!" << std::endl;
}
