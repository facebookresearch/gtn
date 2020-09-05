#define CATCH_CONFIG_MAIN

#include <cmath>
#include <iostream>

#include "catch.hpp"

#include "gtn/functions.h"
#include "gtn/graph.h"
#include "gtn/rand.h"
#include "gtn/utils.h"

using namespace gtn;

TEST_CASE("Test sample", "[rand.sample]") {
  {
    // Sampling empty paths
    Graph empty;

    Graph g;
    CHECK(equal(sample(g), empty));

    g.addNode();
    CHECK(equal(sample(g), empty));

    g.addNode(false, true);
    CHECK(equal(sample(g), empty));

    g.addNode(true);
    CHECK(equal(sample(g), empty));

    g.addArc(0, 0, 1, 0);
    g.addArc(1, 1, 1, 0);
    g.addArc(2, 2, 1, 0);
    CHECK(equal(sample(g), empty));

    g.addArc(0, 2, 0, 1);
    g.addArc(2, 0, 2, 0);
    CHECK(equal(sample(g), empty));

    g.addArc(0, 1, 1, 1);
    CHECK(!equal(sample(g), empty));
  }

  {
    // Check that we can sample the empty path (not the empty graph).
    Graph g;
    g.addNode(true, true);
    CHECK(equal(g, sample(g)));
  }

  {
    // Check some short paths are correct
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 1);
    g.addArc(1, 0, 1);

    auto path = sample(g, 1);
    auto expected =
        loadTxt(std::stringstream("0\n"
                               "1\n"
                               "0 1 1\n"));
    CHECK(equal(expected, path));

    for (int i = 1; i < 20; i++) {
      path = sample(g, i);
      CHECK(!equal(path, Graph{}));
      CHECK(path.numArcs() <= i);
    }
  }
}

TEST_CASE("Test randEquivalent", "[rand.randEquivalent]") {
  {
    // No accepting paths in the graphs
    Graph g1;
    g1.addNode(true);
    g1.addArc(0, 0, 1, 2, 3);
    g1.addNode(false);

    Graph g2;
    g2.addNode();

    Graph g3;
    g3.addNode(true);
    g3.addNode(false, true);
    g3.addNode();
    g3.addArc(0, 2, 1, 1, 4);
    g3.addArc(2, 0, 1, 0, 1);
    g3.addArc(1, 0, 0, 0, 2);

    CHECK(randEquivalent(g1, g2, 100));
    CHECK(randEquivalent(g1, g3, 100));
    CHECK(randEquivalent(g2, g3, 100));
  }

  {
    // Simple not equivalent
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, 1, 0, 1.0);
    g1.addArc(1, 2, 0, 1, 1.0);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    CHECK(!randEquivalent(g1, g2, 100));

    g2.addNode(false, true);
    g2.addArc(0, 1, 1, 1, 1.0);
    g2.addArc(1, 2, 1, 1, 1.0);
    CHECK(!randEquivalent(g1, g2, 100));
  }

  {
    // Simple equivalent
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, 1, 0.5);
    g1.addArc(1, 2, 0, 1, 1.5);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 1, 1.5);
    g2.addArc(1, 2, 0, 1, 0.5);
    CHECK(randEquivalent(g1, g2, 100));
  }

  {
    // Harder not equivalent
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    for (int i = 0; i < 5; i++) {
      g1.addArc(0, 1, i, 5 * i);
    }
    g1.addArc(1, 2, 5, 10);
    g1.addArc(1, 2, 6, 10);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    for (int i = 0; i < 3; i++) {
      g2.addArc(0, 1, i, 5 * i);
    }
    g2.addArc(0, 1, 3, 9);
    g2.addArc(0, 1, 4, 19);
    g2.addArc(1, 2, 5, 10);
    g2.addArc(1, 2, 6, 10);

    CHECK(!randEquivalent(g1, g2, 500));
  }

  {
    // With epsilons
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode();
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, 3);
    g1.addArc(1, 2, 1, 4);
    g1.addArc(2, 3, 2, Graph::epsilon);
    g1.addArc(3, 4, 3, 0);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode();
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 3);
    g2.addArc(1, 2, 1, Graph::epsilon);
    g2.addArc(2, 3, 2, 4);
    g2.addArc(3, 4, 3, 0);

    CHECK(randEquivalent(g1, g2, 100));

    Graph g3;
    g3.addNode(true);
    g3.addNode();
    g3.addNode();
    g3.addNode();
    g3.addNode();
    g3.addNode(false, true);
    g3.addArc(0, 1, 0, 3);
    g3.addArc(1, 2, 1, Graph::epsilon);
    g3.addArc(2, 3, Graph::epsilon, 4);
    g3.addArc(3, 4, 2, Graph::epsilon);
    g3.addArc(4, 5, 3, 0);

    CHECK(randEquivalent(g1, g3, 100));
    CHECK(randEquivalent(g2, g3, 100));
  }
}
