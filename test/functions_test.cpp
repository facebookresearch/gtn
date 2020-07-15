#define CATCH_CONFIG_MAIN

#include <cmath>
#include <iostream>

#include "catch.hpp"

#include "gtn/functions.h"
#include "gtn/graph.h"
#include "gtn/rand.h"
#include "gtn/utils.h"

using namespace gtn;

TEST_CASE("Test Scalar Ops", "[functions.scalars]") {
  Graph g1;
  g1.addNode(true);
  g1.addNode(false, true);
  g1.addArc(0, 1, 0, 0, 3.0);

  auto result = negate(g1);
  CHECK(result.item() == -3.0);

  Graph g2;
  g2.addNode(true);
  g2.addNode(false, true);
  g2.addArc(0, 1, 0, 0, 4.0);

  result = add(g1, g2);
  CHECK(result.item() == 7.0);

  result = subtract(g2, g1);
  CHECK(result.item() == 1.0);
}

TEST_CASE("Test Project/clone", "[functions.clone]") {
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

  // Test clone
  Graph cloned = clone(graph);
  CHECK(equals(graph, cloned));

  // Test projecting input
  Graph inputExpected =
      load(std::stringstream("0 1\n"
                             "3 4\n"
                             "0 1 0 0 2\n"
                             "0 2 1 1 1\n"
                             "1 2 0 0 2\n"
                             "2 3 0 0 1\n"
                             "2 3 1 1 1\n"
                             "1 4 0 0 2\n"
                             "2 4 1 1 3\n"
                             "3 4 0 0 2\n"));
  CHECK(equals(projectInput(graph), inputExpected));

  // Test projecting output
  Graph outputExpected =
      load(std::stringstream("0 1\n"
                             "3 4\n"
                             "0 1 2 2 2\n"
                             "0 2 3 3 1\n"
                             "1 2 1 1 2\n"
                             "2 3 0 0 1\n"
                             "2 3 2 2 1\n"
                             "1 4 1 1 2\n"
                             "2 4 1 1 3\n"
                             "3 4 2 2 2\n"));
  CHECK(equals(projectOutput(graph), outputExpected));
}

TEST_CASE("Test Composition", "[functions.compose]") {
  {
    // Composing with an empty graph gives an empty graph
    Graph g1;
    Graph g2;
    CHECK(equals(compose(g1, g2), Graph{}));

    g1.addNode(true);
    g1.addNode();
    g1.addArc(0, 1, 0);

    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 0);
    g2.addArc(0, 1, 0);

    CHECK(equals(compose(g1, g2), Graph{}));
    CHECK(equals(compose(g2, g1), Graph{}));
    CHECK(equals(intersect(g2, g1), Graph{}));

    // Check singly sorted version
    g1.arcSort(true);
    CHECK(equals(compose(g1, g2), Graph{}));

    // Check doubly sorted version
    g2.arcSort();
    CHECK(equals(compose(g1, g2), Graph{}));
  }

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
    Graph expected = load(in);
    CHECK(isomorphic(compose(g1, g2), expected));
    CHECK(isomorphic(intersect(g1, g2), expected));

    // Check singly sorted version
    g1.arcSort(true);
    CHECK(isomorphic(compose(g1, g2), expected));

    // Check doubly sorted version
    g2.arcSort();
    CHECK(isomorphic(compose(g1, g2), expected));
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
    Graph expected = load(in);
    CHECK(isomorphic(compose(g1, g2), expected));
    CHECK(isomorphic(intersect(g1, g2), expected));

    // Check singly sorted version
    g1.arcSort(true);
    CHECK(isomorphic(compose(g1, g2), expected));

    // Check doubly sorted version
    g2.arcSort();
    CHECK(isomorphic(compose(g1, g2), expected));
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
        g1.addArc(i, i + 1, j, j, j);
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
    Graph expected = load(in);
    CHECK(isomorphic(compose(g1, g2), expected));
    CHECK(isomorphic(intersect(g1, g2), expected));

    // Check singly sorted version
    g1.arcSort(true);
    CHECK(isomorphic(compose(g1, g2), expected));

    // Check doubly sorted version
    g2.arcSort();
    CHECK(isomorphic(compose(g1, g2), expected));
  }
}

TEST_CASE("Test Forward", "[functions.forwardScore]") {
  {
    // Throws on self-loops
    Graph g;
    g.addNode(true, true);
    g.addArc(0, 0, 1);
    CHECK_THROWS(forwardScore(g));
  }

  {
    // Throws on internal self-loop
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0);
    g.addArc(1, 2, 0);
    g.addArc(1, 1, 0);
    CHECK_THROWS(forwardScore(g));
  }

  {
    // Throws on self-loop in accept node
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0);
    g.addArc(1, 2, 0);
    g.addArc(2, 2, 0);
    CHECK_THROWS(forwardScore(g));
  }

  {
    // Throws on cycle
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0);
    g.addArc(1, 2, 0);
    g.addArc(2, 0, 0);
    CHECK_THROWS(forwardScore(g));
  }

  {
    // Throws if a non-start node has no incoming arcs
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 2, 0);
    g.addArc(1, 2, 0);
    CHECK_THROWS(forwardScore(g));
  }

  {
    // A simple test case
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 1);
    g.addArc(0, 1, 1, 1, 2);
    g.addArc(0, 1, 2, 2, 3);
    g.addArc(1, 2, 0, 0, 1);
    g.addArc(1, 2, 1, 1, 2);
    g.addArc(1, 2, 2, 2, 3);
    CHECK(forwardScore(g).item() == Approx(6.8152));
  }

  {
    // Handle two start nodes
    Graph g;
    g.addNode(true);
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, -5);
    g.addArc(0, 2, 0, 0, 1);
    g.addArc(1, 2, 0, 0, 2);
    float expected = std::log(std::exp(1) + std::exp(-5 + 2) + std::exp(2));
    CHECK(forwardScore(g).item() == Approx(expected));
  }

  {
    // Handle two accept nodes
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 2);
    g.addArc(0, 2, 0, 0, 2);
    g.addArc(1, 2, 0, 0, 2);
    float expected = std::log(2 * std::exp(2) + std::exp(4));
    CHECK(forwardScore(g).item() == Approx(expected));
  }

  {
    // A more complex test case
    std::stringstream in(
        "0 1\n"
        "3 4\n"
        "0 1 0 0 2\n"
        "0 2 1 1 1\n"
        "1 2 0 0 2\n"
        "2 3 0 0 1\n"
        "2 3 1 1 1\n"
        "1 4 0 0 2\n"
        "2 4 1 1 3\n"
        "3 4 0 0 2\n");
    Graph g = load(in);
    CHECK(forwardScore(g).item() == Approx(8.36931));
  }
}

TEST_CASE("Test Viterbi Score", "[functions.viterbiScore]") {
  {
    // A simple test case
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 1);
    g.addArc(0, 1, 1, 1, 2);
    g.addArc(0, 1, 2, 2, 3);
    g.addArc(1, 2, 0, 0, 1);
    g.addArc(1, 2, 1, 1, 2);
    g.addArc(1, 2, 2, 2, 3);
    CHECK(viterbiScore(g).item() == 6.0f);
  }

  {
    // Handle two start nodes
    Graph g;
    g.addNode(true);
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, -5);
    g.addArc(0, 2, 0, 0, 1);
    g.addArc(1, 2, 0, 0, 2);
    CHECK(viterbiScore(g).item() == 2.0f);
  }

  {
    // Handle two accept nodes
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 2);
    g.addArc(0, 2, 0, 0, 2);
    g.addArc(1, 2, 0, 0, 2);
    CHECK(viterbiScore(g).item() == 4.0f);
  }

  {
    // A more complex test case
    std::stringstream in(
        "0 1\n"
        "3 4\n"
        "0 1 0 0 2\n"
        "0 2 1 1 1\n"
        "1 2 0 0 2\n"
        "2 3 0 0 1\n"
        "2 3 1 1 1\n"
        "1 4 0 0 2\n"
        "2 4 1 1 3\n"
        "3 4 0 0 2\n");
    Graph g = load(in);
    CHECK(viterbiScore(g).item() == 7.0f);
  }
}

TEST_CASE("Test Viterbi Path", "[functions.viterbiPath]") {
  {
    // A simple test case
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 1);
    g.addArc(0, 1, 1, 1, 2);
    g.addArc(0, 1, 2, 2, 3);
    g.addArc(1, 2, 0, 0, 1);
    g.addArc(1, 2, 1, 1, 2);
    g.addArc(1, 2, 2, 2, 3);

    Graph best;
    best.addNode(true);
    best.addNode();
    best.addNode(false, true);
    best.addArc(0, 1, 2, 2, 3);
    best.addArc(1, 2, 2, 2, 3);

    auto path = viterbiPath(g);
    CHECK(randEquivalent(path, best));
    CHECK(viterbiScore(path).item() == viterbiScore(g).item());
  }

  {
    // Handle a single node.
    Graph g;
    g.addNode(true, true);

    Graph best;
    best.addNode(true, true);
    auto path = viterbiPath(g);
    CHECK(randEquivalent(path, best));
    CHECK(viterbiScore(path).item() == viterbiScore(g).item());
  }

  {
    // Handle two start nodes
    Graph g;
    g.addNode(true);
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, -5);
    g.addArc(0, 2, 0, 0, 1);
    g.addArc(1, 2, 0, 0, 2);

    Graph best;
    best.addNode(true);
    best.addNode(false, true);
    best.addArc(0, 1, 0, 0, 2);

    auto path = viterbiPath(g);
    CHECK(randEquivalent(path, best));
    CHECK(viterbiScore(path).item() == viterbiScore(g).item());
  }

  {
    // Handle two accept nodes
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 3);
    g.addArc(0, 2, 0, 0, 2);
    g.addArc(1, 2, 0, 0, 2);

    Graph best;
    best.addNode(true);
    best.addNode();
    best.addNode(false, true);
    best.addArc(0, 1, 0, 0, 3);
    best.addArc(1, 2, 0, 0, 2);

    auto path = viterbiPath(g);
    CHECK(randEquivalent(path, best));
    CHECK(viterbiScore(path).item() == viterbiScore(g).item());
  }

  {
    // A more complex test case
    std::stringstream in(
        "0 1\n"
        "3 4\n"
        "0 1 0 0 2\n"
        "0 2 1 1 1\n"
        "1 2 0 0 2\n"
        "2 3 0 0 1\n"
        "2 3 1 1 1\n"
        "1 4 0 0 2\n"
        "2 4 1 1 3\n"
        "3 4 0 0 2\n");
    Graph g = load(in);

    // There are three options for the best path, the
    // viterbiPath may return any of them.
    Graph best1;
    best1.addNode(true);
    best1.addNode();
    best1.addNode();
    best1.addNode();
    best1.addNode(false, true);
    best1.addArc(0, 1, 0, 0, 2);
    best1.addArc(1, 2, 0, 0, 2);
    best1.addArc(2, 3, 0, 0, 1);
    best1.addArc(3, 4, 0, 0, 2);

    Graph best2;
    best2.addNode(true);
    best2.addNode();
    best2.addNode();
    best2.addNode();
    best2.addNode(false, true);
    best2.addArc(0, 1, 0, 0, 2);
    best2.addArc(1, 2, 0, 0, 2);
    best2.addArc(2, 3, 1, 1, 1);
    best2.addArc(3, 4, 0, 0, 2);

    Graph best3;
    best3.addNode(true);
    best3.addNode();
    best3.addNode();
    best3.addNode(false, true);
    best3.addArc(0, 1, 0, 0, 2);
    best3.addArc(1, 2, 0, 0, 2);
    best3.addArc(2, 3, 1, 1, 3);

    auto path = viterbiPath(g);
    CHECK(
        (randEquivalent(path, best1) || randEquivalent(path, best2) ||
         randEquivalent(path, best3)));

    CHECK(viterbiScore(path).item() == viterbiScore(g).item());
  }
}

TEST_CASE("Test Epsilon Composition", "[functions.epsilon_compose]") {
  {
    // Simple test case for output epsilon on first graph
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 0, 0, Graph::epsilon, 1.0);
    g1.addArc(0, 1, 1, 2);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 2, 3);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addArc(0, 0, 0, Graph::epsilon, 1.0);
    expected.addArc(0, 1, 1, 3);

    CHECK(equals(compose(g1, g2), expected));
  }

  {
    // Simple test case for input epsilon on second graph
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 1, 2);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 2, 3);
    g2.addArc(1, 1, Graph::epsilon, 0, 2.0);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addArc(0, 1, 1, 3);
    expected.addArc(1, 1, Graph::epsilon, 0, 2.0);

    CHECK(equals(compose(g1, g2), expected));
  }

  {
    // This test case is taken from "Weighted Automata Algorithms", Mehryar
    // Mohri, https://cs.nyu.edu/~mohri/pub/hwa.pdf Section 5.1, Figure 7
    std::unordered_map<std::string, int> symbols = {
        {"a", 0}, {"b", 1}, {"c", 2}, {"d", 3}, {"e", 4}};
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode();
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, symbols["a"], symbols["a"]);
    g1.addArc(1, 2, symbols["b"], Graph::epsilon);
    g1.addArc(2, 3, symbols["c"], Graph::epsilon);
    g1.addArc(3, 4, symbols["d"], symbols["d"]);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, symbols["a"], symbols["d"]);
    g2.addArc(1, 2, Graph::epsilon, symbols["e"]);
    g2.addArc(2, 3, symbols["d"], symbols["a"]);

    Graph expected;
    expected.addNode(true);
    expected.addNode();
    expected.addNode();
    expected.addNode();
    expected.addNode(false, true);
    expected.addArc(0, 1, symbols["a"], symbols["d"]);
    expected.addArc(1, 2, symbols["b"], symbols["e"]);
    expected.addArc(2, 3, symbols["c"], Graph::epsilon);
    expected.addArc(3, 4, symbols["d"], symbols["a"]);

    CHECK(randEquivalent(compose(g1, g2), expected));
  }

  {
    // Test multiple input/output epsilon transitions per node
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 0, 1, Graph::epsilon, 1.1);
    g1.addArc(0, 1, 2, Graph::epsilon, 2.1);
    g1.addArc(0, 1, 3, Graph::epsilon, 3.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, Graph::epsilon, 3, 2.1);
    g2.addArc(0, 1, 1, 2);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addArc(0, 0, 1, Graph::epsilon, 1.1);
    expected.addArc(0, 1, 2, 3, 4.2);
    expected.addArc(0, 1, 3, 3, 5.2);

    CHECK(randEquivalent(compose(g1, g2), expected));
  }
}

TEST_CASE("Test Concat", "[functions.concat]") {
  {
    // Empty string language
    Graph g;
    g.addNode(true, true);

    CHECK(equals(concat({}), g));
    CHECK(randEquivalent(concat(g, g), g));
    CHECK(randEquivalent(concat({g, g, g}), g));
  }
  {
    // Singleton
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 1);
    CHECK(equals(concat({g}), g));
  }

  {
    // Empty language
    Graph g;
    g.addNode();
    CHECK(randEquivalent(concat(g, g), Graph{}));
    CHECK(randEquivalent(concat({g, g, g}), Graph{}));
  }

  {
    // Concat {0} and {1} to get {01}
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 1);

    Graph expected;
    expected.addNode(true);
    expected.addNode();
    expected.addNode(false, true);
    expected.addArc(0, 1, 0);
    expected.addArc(1, 2, 1);

    CHECK(randEquivalent(concat(g1, g2), expected));
  }

  {
    // Concat {0, 1} and {2, 3} to get {02, 03, 12, 13}
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0);
    g1.addArc(0, 2, 1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 2, 2);
    g2.addArc(1, 2, 3);

    Graph expected;
    expected.addNode(true);
    expected.addNode();
    expected.addNode(false, true);
    expected.addArc(0, 1, 0);
    expected.addArc(0, 1, 1);
    expected.addArc(1, 2, 2);
    expected.addArc(1, 2, 3);

    CHECK(randEquivalent(concat(g1, g2), expected));
  }
}

TEST_CASE("Test Closure", "[functions.closure]") {
  {
    // Empty graph
    Graph expected;
    expected.addNode(true, true);
    CHECK(equals(closure(Graph{}), expected));
  }

  {
    // Multi-start, multi-accept
    Graph g;
    g.addNode(true);
    g.addNode(true);
    g.addNode(false, true);
    g.addNode(false, true);
    g.addArc(0, 2, 0, 0, 0.0);
    g.addArc(0, 3, 1, 2, 2.1);
    g.addArc(1, 2, 0, 1, 1.0);
    g.addArc(1, 3, 1, 3, 3.1);

    Graph expected;
    expected.addNode(true, true);
    expected.addNode();
    expected.addNode();
    expected.addNode(false, true);
    expected.addNode(false, true);
    expected.addArc(0, 1, Graph::epsilon);
    expected.addArc(0, 2, Graph::epsilon);
    expected.addArc(1, 3, 0, 0, 0.0);
    expected.addArc(1, 4, 1, 2, 2.1);
    expected.addArc(2, 3, 0, 1, 1.0);
    expected.addArc(2, 4, 1, 3, 3.1);
    expected.addArc(3, 1, Graph::epsilon);
    expected.addArc(3, 2, Graph::epsilon);
    expected.addArc(4, 1, Graph::epsilon);
    expected.addArc(4, 2, Graph::epsilon);

    CHECK(randEquivalent(closure(g), expected));
  }
}

TEST_CASE("Test Sum", "[functions.sum]") {
  {
    // Empty graph
    CHECK(equals(sum({}), Graph{}));
  }

  {
    // Check single graph is a no-op
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 1);
    CHECK(equals(sum({g1}), g1));
  }

  {
    // Simple sum
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 0);

    Graph expected;
    expected.addNode(true);
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addNode(false, true);
    expected.addArc(0, 2, 1);
    expected.addArc(1, 3, 0);
    CHECK(isomorphic(sum({g1, g2}), expected));
  }

  {
    // Check adding with an empty graph works
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 1);

    Graph g2;

    Graph g3;
    g3.addNode(true, true);
    g3.addArc(0, 0, 2);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addNode(true, true);
    expected.addArc(0, 1, 1);
    expected.addArc(2, 2, 2);
    CHECK(isomorphic(sum({g1, g2, g3}), expected));
  }
}

TEST_CASE("Test Remove", "[functions.remove]") {
  {
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, Graph::epsilon);
    g.addArc(1, 2, 0);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addArc(0, 1, 0);
    CHECK(equals(remove(g, Graph::epsilon), expected));
  }

  {
    // Removing other labels works
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 2, 1);
    g.addArc(1, 2, 0, 1);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addArc(0, 1, 0, 1);
    CHECK(equals(remove(g, 2, 1), expected));
  }

  {
    // No-op on graph without epsilons
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 1);
    g.addArc(0, 1, 1, 1);
    CHECK(equals(remove(g), g));
  }

  {
    // Epsilon only transitions into accepting state
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0);
    g.addArc(0, 2, 1);
    g.addArc(1, 3, Graph::epsilon);
    g.addArc(2, 3, Graph::epsilon);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addNode(false, true);
    expected.addArc(0, 1, 0);
    expected.addArc(0, 2, 1);
    CHECK(equals(remove(g), expected));
  }

  {
    // Only remove an arc, no removed nodes
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, Graph::epsilon);
    g.addArc(0, 2, 1);
    g.addArc(2, 1, 0);
    g.addArc(1, 3, 1);

    Graph expected;
    expected.addNode(true);
    expected.addNode();
    expected.addNode();
    expected.addNode(false, true);
    expected.addArc(0, 2, 1);
    expected.addArc(2, 1, 0);
    expected.addArc(1, 3, 1);
    expected.addArc(0, 3, 1);
    CHECK(equals(remove(g), expected));
  }

  {
    // Successive epsilons
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode();
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0);
    g.addArc(1, 2, Graph::epsilon);
    g.addArc(2, 3, Graph::epsilon);
    g.addArc(2, 4, 1);
    g.addArc(3, 4, 2);
    g.addArc(1, 4, 0);

    Graph expected;
    expected.addNode(true);
    expected.addNode();
    expected.addNode(false, true);
    expected.addArc(0, 1, 0);
    expected.addArc(1, 2, 0);
    expected.addArc(1, 2, 1);
    expected.addArc(1, 2, 2);
    CHECK(equals(remove(g), expected));
  }

  // Multiple interior removals
  {
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode();
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, Graph::epsilon);
    g.addArc(1, 2, Graph::epsilon);
    g.addArc(2, 3, 0);
    g.addArc(3, 4, 0);

    Graph expected;
    expected.addNode(true);
    expected.addNode();
    expected.addNode(false, true);
    expected.addArc(0, 1, 0);
    expected.addArc(1, 2, 0);
    CHECK(equals(remove(g), expected));
  }
}
