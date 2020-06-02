#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "gtn/graph.h"

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

  g.addArc(1, 1, 1);
  g.addArc(2, 3, 2);
  CHECK(g.numArcs() == 5);
  CHECK(g.node(0)->numOut() == 2);
  CHECK(g.node(1)->numIn() == 2);

  CHECK_THROWS(g.addArc(0, 5, 0));
  CHECK_THROWS(g.addArc(-1, 3, 0));
  CHECK_THROWS(g.addArc(5, 0, 0));

  CHECK_THROWS(g.node(5));

  // If we copy the graph it should have the same structure.
  Graph g_copy = g;
  CHECK(g_copy.numNodes() == 5);
  CHECK(g_copy.numStart() == 1);
  CHECK(g_copy.numAccept() == 1);
  CHECK(g_copy.numArcs() == 5);
  CHECK(g_copy.node(0)->numOut() == 2);
  CHECK(g_copy.node(1)->numIn() == 2);

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
  CHECK(g2.arcs()[0].label() == 0);
  CHECK(g2.arcs()[0].downNode()->index() == 1);
  CHECK(g2.arcs()[0].upNode()->index() == 0);

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
    // Check adding arcs using pointers.
    Graph g;
    auto n1 = g.addNode(true);
    auto n2 = g.addNode();
    g.addArc(n1, n2, 1, 1, 3.0);
    CHECK(g.arcs()[0].label() == 1);
    CHECK(g.item() == 3.0f);
  }

  {
    // Check adding transducing arcs
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    auto a1 = g.addArc(0, 1, 1, 2);
    CHECK(g.arcs()[0].ilabel() == 1);
    CHECK(g.arcs()[0].olabel() == 2);
    auto a2 = g.addArc(0, 1, 1, 0, 2);
    CHECK(g.arcs()[1].ilabel() == 1);
    CHECK(g.arcs()[1].olabel() == 0);
  }

  {
    // Check acceptor status
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    CHECK(g.acceptor());
    g.addArc(0, 1, 1);
    g.addArc(1, 2, 1, 1);
    g.addArc(0, 1, 0, 0, 3.0);
    CHECK(g.acceptor());
    g.addArc(1, 2, 2, 1);
    g.addArc(1, 2, 2, 2);
    CHECK(!g.acceptor());
  }
}
