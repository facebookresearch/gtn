#define CATCH_CONFIG_MAIN

#include <algorithm>
#include <cstdlib>

#include "catch.hpp"

#include "gtn/graph.h"
#include "gtn/utils.h"

using namespace gtn;

// Override globals just for testing
size_t allocations;
size_t deallocations;

void* operator new(std::size_t size) {
  allocations++;
  return std::malloc(size);
}

void operator delete(void* p) throw() {
  deallocations++;
  free(p);
}

TEST_CASE("Test Graph", "[graph]") {
  Graph g;
  g.addNode(true);
  g.addNode();
  g.addNode();
  g.addNode();
  g.addNode(false, true);
  CHECK(g.numNodes() == 5);
  CHECK(g.numStart() == 1);
  CHECK(g.numAccept() == 1);

  g.addArc(0, 1, 0);
  g.addArc(0, 2, 1);
  g.addArc(1, 2, 0);
  g.addArc(1, 1, 1, 1, 2.1);
  g.addArc(2, 3, 2);

  CHECK(g.numArcs() == 5);
  CHECK(g.numOut(0) == 2);
  CHECK(g.numIn(1) == 2);

  // If we (shallow) copy the graph it should have the same structure.
  Graph g_copy = g;
  CHECK(g_copy.numNodes() == 5);
  CHECK(g_copy.numStart() == 1);
  CHECK(g_copy.numAccept() == 1);
  CHECK(g_copy.numArcs() == 5);
  CHECK(g_copy.numOut(0) == 2);
  CHECK(g_copy.numIn(1) == 2);
  CHECK(g_copy.weight(3) == 2.1f);

  // If we construct a graph from another graph it should also have the same
  // structure.
  Graph g_copy2 = Graph(g, nullptr, {});
  CHECK(equal(g_copy2, g));

  // Modifying g should modify g_copy and g_copy2
  g.addNode();
  g.addNode();
  g.addNode();
  g.addNode();
  for (int i = 0; i < 8; i++) {
    g.addArc(i, i, i);
    g.addArc(i, i, i + 1);
    g.addArc(i, i + 1, i);
    g.addArc(i, i + 1, i + 1);
  }
  CHECK(equal(g, g_copy));
  CHECK(equal(g, g_copy2));

  // Check that we can copy a graph and the destination still
  // works when the source graph is out of scope
  Graph g2;
  {
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addArc(0, 1, 0);
    g2 = g1;
  }
  CHECK(g2.numNodes() == 2);
  CHECK(g2.label(0) == 0);
  CHECK(g2.downNode(0) == 1);
  CHECK(g2.upNode(0) == 0);

  {
    // We can get a scalar out of a single arc graph.
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 1, 1, 3.3);
    CHECK(g.item() == 3.3f);

    // We cannot get a scalar out of a many arc graph.
    g.addArc(0, 1, 2, 2, 3.3);
    CHECK_THROWS(g.item());
  }

  allocations = 0;
  deallocations = 0;
  // We should see flat memory use
  {
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 1);
    g.addArc(0, 0, 0);
  }
  CHECK(allocations == deallocations);

  {
    // Check adding transducing arcs
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 1, 2);
    CHECK(g.ilabel(0) == 1);
    CHECK(g.olabel(0) == 2);
    g.addArc(0, 1, 1, 0, 2);
    CHECK(g.ilabel(1) == 1);
    CHECK(g.olabel(1) == 0);
  }
}

TEST_CASE("Test copy", "[Graph::deepCopy]") {
  Graph graph =
      load(std::stringstream("0 1\n"
                             "3 4\n"
                             "0 1 0 2 2\n"
                             "0 2 1 3 1\n"
                             "1 2 0 1 2\n"
                             "2 3 0 0 1\n"
                             "2 3 1 2 1\n"
                             "1 4 0 1 2\n"
                             "2 4 1 1 3\n"
                             "3 4 0 2 2\n"));

  // Test copy
  Graph copied = Graph::deepCopy(graph);
  CHECK(equal(copied, graph));
  CHECK(copied.calcGrad() == graph.calcGrad());
  CHECK(copied.id() != graph.id());

  copied.addArc(0, 3, 0);
  CHECK(!equal(copied, graph));
}

TEST_CASE("Test arc weight get/set", "[graph setWeight weights]") {
  std::vector<float> l = {1.1, 2.2, 3.3, 4.4};

  Graph g;
  g.addNode(true, false);
  g.addNode();
  g.addNode();
  g.addNode();
  g.addNode(false, true);
  g.addArc(0, 1, 0);
  g.addArc(1, 2, 0);
  g.addArc(2, 3, 0);
  g.addArc(3, 4, 0);
  g.setWeights(l.data());

  CHECK(l == std::vector<float>(g.weights(), g.weights() + g.numArcs()));
}

TEST_CASE("Test arc label getters", "[graph labelsToVector]") {
  std::vector<int> l = {0, 1, 2, 3};

  Graph g;
  g.addNode(true, false);
  g.addNode();
  g.addNode();
  g.addNode(false, true);
  g.addArc(0, 1, l[0], l[3]);
  g.addArc(0, 2, l[1], l[2]);
  g.addArc(1, 2, l[2], l[1]);
  g.addArc(1, 3, l[3], l[0]);

  CHECK(l == g.labelsToVector(/*ilabel=*/true));
  std::reverse(l.begin(), l.end());
  CHECK(l == g.labelsToVector(/*ilabel=*/false));
}

TEST_CASE("Test gradient functionality", "[graph grad]") {
  {
    // calcGrad is false
    Graph g(false);
    CHECK_THROWS(g.grad());

    g.addGrad(Graph{});
    CHECK_THROWS(g.grad());

    g.addNode();
    g.addArc(0, 0, 0);
    g.addArc(0, 0, 1);

    // this should be a no-op
    g.addGrad({0, 0, 0});
    CHECK_THROWS(g.grad());
  }

  {
    // No gradient yet
    Graph g(true);
    CHECK_THROWS(g.grad());

    // Empty gradient
    g.addGrad(Graph{});
    CHECK(equal(g.grad(), Graph{}));

    // No gradient
    g.zeroGrad();
    CHECK_THROWS(g.grad());

    g.addNode();
    g.addNode();
    g.addArc(0, 1, 0);

    Graph grad;
    grad.addNode();
    grad.addNode();
    grad.addArc(0, 1, 0, 0, 1.0);

    g.addGrad(grad);
    CHECK(equal(g.grad(), grad));

    // Grads accumulate properly
    g.addGrad(grad);
    Graph expected;
    expected.addNode();
    expected.addNode();
    expected.addArc(0, 1, 0, 0, 2.0);
    CHECK(equal(g.grad(), expected));

    // Wrong sized grad throws
    grad.addArc(0, 1, 0, 0, 2.0);
    CHECK_THROWS(g.addGrad(grad));
    CHECK_THROWS(g.addGrad({0.0, 1.0}));
  }

  {
    // calcGrad propgates properly
    Graph g1(true);
    Graph g2(false);
    Graph g3(nullptr, {g1, g2});
    CHECK(g3.calcGrad());

    Graph g4(false);
    Graph g5(nullptr, {g2, g4});
    CHECK(!g5.calcGrad());
  }

  {
    // Check copy vs move
    Graph g;
    g.addNode();
    g.addNode();
    g.addArc(0, 1, 0);

    auto grad = Graph::deepCopy(g);
    grad.setWeight(0, 2.0);

    // this should make a copy of grads weights
    g.addGrad(grad);
    grad.setWeight(0, 4.0);
    CHECK(g.grad().weight(0) == 2.0f);

    // this should make a copy
    std::vector<float> gradsV = {1.0};
    g.addGrad(gradsV);
    g.grad().setWeight(0, 2.0);
    CHECK(gradsV[0] == 1.0);
  }
}

TEST_CASE("Test sort", "[Graph::arcSort]") {
  // sort on empty graph does nothing
  Graph g;
  g.arcSort();

  g.addNode();
  g.addNode();
  g.addNode();

  g.addArc(0, 1, 1, 3);
  g.addArc(0, 1, 0, 2);
  g.addArc(0, 1, 3, 4);
  g.addArc(1, 1, 3, 0);
  g.addArc(1, 1, 0, 4);
  g.addArc(1, 2, 0, 4);
  g.addArc(1, 2, 1, 1);
  g.addArc(1, 2, 2, 0);

  // sort on ilabel
  g.arcSort();
  auto ilabelCmp = [&g](int a, int b) { return g.ilabel(a) < g.ilabel(b); };
  for (auto n = 0; n < g.numNodes(); ++n) {
    CHECK(std::is_sorted(g.in(n).begin(), g.in(n).end(), ilabelCmp));
    CHECK(std::is_sorted(g.out(n).begin(), g.out(n).end(), ilabelCmp));
  }
  CHECK(g.ilabelSorted());
  CHECK(!g.olabelSorted());

  // sort on olabel
  g.arcSort(true);
  auto olabelCmp = [&g](int a, int b) { return g.olabel(a) < g.olabel(b); };
  for (auto n = 0; n < g.numNodes(); ++n) {
    CHECK(std::is_sorted(g.in(n).begin(), g.in(n).end(), olabelCmp));
    CHECK(std::is_sorted(g.out(n).begin(), g.out(n).end(), olabelCmp));
  }
  CHECK(!g.ilabelSorted());
  CHECK(g.olabelSorted());

  g.markArcSorted();
  CHECK(g.ilabelSorted());

  g.addArc(1, 2, 0, 3);
  CHECK(!g.olabelSorted());

  g.markArcSorted(true);
  CHECK(g.olabelSorted());
}
