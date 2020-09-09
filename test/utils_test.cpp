#define CATCH_CONFIG_MAIN

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "catch.hpp"

#include "gtn/graph.h"
#include "gtn/utils.h"

using namespace gtn;

namespace {
std::string getTmpFileName() {
  // Get user unique filename
  char* user = std::getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  return std::string("/tmp/test_") + userstr + std::string(".graph");
}
} // namespace

TEST_CASE("Test Graph equality", "[utils.equal]") {
  {
    // Empty graph is equal to itself
    Graph g1;
    Graph g2;
    CHECK(equal(g1, g2));
  }

  {
    // Different start node
    Graph g1;
    g1.addNode(true);

    Graph g2;
    g2.addNode(false);
    CHECK_FALSE(equal(g1, g2));
  }

  {
    // Simple equality
    Graph g1;
    g1.addNode(true);
    g1.addNode();

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    CHECK(equal(g1, g2));
  }

  {
    // Different arc label
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addArc(0, 1, 0);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addArc(0, 1, 1);
    CHECK_FALSE(equal(g1, g2));
  }

  {
    // Different arc weight
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addArc(0, 1, 0, 0, 1.2);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addArc(0, 1, 0, 0, 2.2);
    CHECK_FALSE(equal(g1, g2));
  }

  {
    // Self loop in g1
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, 0);
    g1.addArc(0, 1, 1);
    g1.addArc(1, 1, 1);
    g1.addArc(1, 2, 2);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0);
    g2.addArc(0, 1, 1);
    g2.addArc(1, 2, 2);
    CHECK_FALSE(equal(g1, g2));
  }

  {
    // Equals
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, 0, 2.1);
    g1.addArc(0, 1, 1, 1, 3.1);
    g1.addArc(1, 1, 1, 1, 4.1);
    g1.addArc(1, 2, 2, 2, 5.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, 2.1);
    g2.addArc(0, 1, 1, 1, 3.1);
    g2.addArc(1, 1, 1, 1, 4.1);
    g2.addArc(1, 2, 2, 2, 5.1);
    CHECK(equal(g1, g2));
  }

  {
    // Different arc order
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, 1, 1, 3.1);
    g1.addArc(0, 1, 0, 0, 2.1);
    g1.addArc(1, 1, 1, 1, 4.1);
    g1.addArc(1, 2, 2, 2, 5.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, 2.1);
    g2.addArc(0, 1, 1, 1, 3.1);
    g2.addArc(1, 2, 2, 2, 5.1);
    g2.addArc(1, 1, 1, 1, 4.1);
    CHECK(equal(g1, g2));
  }

  {
    // Repeat arcs
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 1, 1, 3.1);
    g1.addArc(0, 1, 1, 1, 3.1);
    g1.addArc(0, 1, 1, 1, 4.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 1, 1, 3.1);
    g2.addArc(0, 1, 1, 1, 4.1);
    g2.addArc(0, 1, 1, 1, 4.1);
    CHECK(!equal(g1, g2));
  }

  {
    // Transducer with different outputs
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, 1, 2.1);
    g1.addArc(1, 1, 1, 3, 4.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 1, 2.1);
    g2.addArc(1, 1, 1, 4, 4.1);
    CHECK(!equal(g1, g2));
  }
}

TEST_CASE("Test Graph isomorphic", "[utils.isomorphic]") {
  {
    Graph g1;
    g1.addNode(true);

    Graph g2;
    g2.addNode(true);

    CHECK(isomorphic(g1, g2));
  }

  {
    Graph g1;
    g1.addNode(true);

    Graph g2;
    g2.addNode(true, true);

    CHECK(!isomorphic(g1, g2));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, 0, 1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, 1);

    CHECK(isomorphic(g1, g2));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, 0, 1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, 1);
    g2.addArc(0, 1, 0, 0, 1);

    CHECK(!isomorphic(g1, g2));
  }

  {
    // Graph with loops
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 2, 0);
    g1.addArc(0, 1, 0);
    g1.addArc(1, 1, 3);
    g1.addArc(2, 1, 3);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addNode();
    g2.addArc(0, 1, 0);
    g2.addArc(1, 2, 3);
    g2.addArc(0, 2, 0);
    g2.addArc(2, 2, 3);

    CHECK(isomorphic(g1, g2));
  }

  {
    // Repeat arcs
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 1, 1, 3.1);
    g1.addArc(0, 1, 1, 1, 3.1);
    g1.addArc(0, 1, 1, 1, 4.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 1, 1, 3.1);
    g2.addArc(0, 1, 1, 1, 4.1);
    g2.addArc(0, 1, 1, 1, 4.1);
    CHECK(!isomorphic(g1, g2));
  }

  {
    // Transducer with different outputs
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, 1, 2.1);
    g1.addArc(1, 1, 1, 3, 4.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 1, 2.1);
    g2.addArc(1, 1, 1, 4, 4.1);
    CHECK(!isomorphic(g1, g2));
  }
}

TEST_CASE("Test loadTxt", "[utils.loadTxt]") {
  {
    Graph g1;
    g1.addNode(true, true);
    g1.addNode(false, true);
    g1.addNode();
    g1.addArc(0, 0, 1);
    g1.addArc(0, 2, 1, 1, 1.1);
    g1.addArc(2, 1, 2, 2, 2.1);

    std::string graph_string =
        "0\n"
        "0 1\n"
        "0 0 1 1 0\n"
        "0 2 1 1 1.1\n"
        "2 1 2 2 2.1\n";
    auto g2 = loadTxt(std::stringstream(graph_string));
    CHECK(equal(g1, g2));
    CHECK(isomorphic(g1, g2));

    auto fn = getTmpFileName();

    // Write the test file
    std::ofstream outf;
    outf.open(fn);
    outf << graph_string;
    outf.close();
    auto g3 = loadTxt(fn);
    CHECK(equal(g1, g3));
  }

  // Empty graph doesn't load
  {
    std::stringstream in("");
    CHECK_THROWS(loadTxt(in));
  }

  // Graph without accept nodes doesn't load
  {
    std::stringstream in("1\n");
    CHECK_THROWS(loadTxt(in));
  }

  // Graph with repeat start nodes doesn't load
  {
    std::stringstream in("1 0 0\n0 1");
    CHECK_THROWS(loadTxt(in));
  }

  // Graph loads if the start and accept nodes are specified
  {
    std::stringstream in("0\n1\n");
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    CHECK(equal(g, loadTxt(in)));
  }

  // Graph doesn't load if arc incorrect
  {
    CHECK_THROWS(loadTxt(std::stringstream("0\n1\n0 2\n")));
    CHECK_THROWS(loadTxt(std::stringstream("0\n1\n0 1 2 3 4 5\n")));
  }

  // Transducer loads
  {
    Graph g1;
    g1.addNode(true, true);
    g1.addNode(false, true);
    g1.addNode();
    g1.addArc(0, 0, 1, 1);
    g1.addArc(0, 2, 1, 2, 1.1);
    g1.addArc(2, 1, 2, 3, 2.1);

    std::string graph_string =
        "0\n"
        "0 1\n"
        "0 0 1\n"
        "0 2 1 2 1.1\n"
        "2 1 2 3 2.1\n";
    auto g2 = loadTxt(std::stringstream(graph_string));
    CHECK(equal(g1, g2));
    CHECK(isomorphic(g1, g2));
  }
}

TEST_CASE("Test SaveTxt", "[utils.saveTxt]") {
  {
    // Acceptor test
    Graph g;
    g.addNode(true);
    g.addNode(true);
    g.addNode();
    g.addNode();
    g.addNode(false, true);
    g.addNode(false, true);

    g.addArc(0, 1, 0, 0, 1.1);
    g.addArc(1, 2, 1, 1, 2.1);
    g.addArc(2, 3, 2, 2, 3.1);
    g.addArc(3, 4, 3, 3, 4.1);
    g.addArc(4, 5, 4, 4, 5.1);

    std::string graph_string =
        "0 1\n"
        "4 5\n"
        "0 1 0 0 1.1\n"
        "1 2 1 1 2.1\n"
        "2 3 2 2 3.1\n"
        "3 4 3 3 4.1\n"
        "4 5 4 4 5.1\n";
    std::stringstream out;
    saveTxt(out, g);
    CHECK(out.str() == graph_string);
  }
  {
    // Transducer test
    Graph g;
    g.addNode(true);
    g.addNode(true);
    g.addNode();
    g.addNode();
    g.addNode(false, true);
    g.addNode(false, true);

    g.addArc(0, 1, 0, 1, 1.1);
    g.addArc(1, 2, 1, 2, 2.1);
    g.addArc(2, 3, 2, 3, 3.1);
    g.addArc(3, 4, 3, 4, 4.1);
    g.addArc(4, 5, 4, Graph::epsilon, 5.1);

    std::string graph_string =
        "0 1\n"
        "4 5\n"
        "0 1 0 1 1.1\n"
        "1 2 1 2 2.1\n"
        "2 3 2 3 3.1\n"
        "3 4 3 4 4.1\n"
        "4 5 4 -1 5.1\n";
    std::stringstream out;
    saveTxt(out, g);
    CHECK(out.str() == graph_string);
  }

  {
    // Save summary
    Graph g;
    for (int i = 0; i < 10; ++i) {
      g.addNode(true, true);
    }
    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 10; ++j) {
        g.addArc(i, j, i, j, i*j);
      }
    }
    std::stringstream out;
    out << g;
    std::string line;
    int numLines = 0;
    while(std::getline(out, line)) {
      CHECK(line.size() < 20);
      ++numLines;
    }
    CHECK(numLines < 20);
  }
}

TEST_CASE("Test loadsave", "[utils.load, utils.save]") {
  auto fn = getTmpFileName();
  {
    Graph g;
    std::stringstream stream;
    save(stream, g);
    std::cout << stream.str();
    Graph g2 = load(stream);
    CHECK(equal(g, g2));

    save(fn, g);
    Graph g3 = load(fn);
    CHECK(equal(g, g3));
  }
  {
    Graph g;
    g.addNode(true);
    g.addNode(true);
    g.addNode();
    g.addNode();
    g.addNode(false, true);
    g.addNode(false, true);

    g.addArc(0, 1, 0, 1, 1.1);
    g.addArc(1, 2, 1, 2, 2.1);
    g.addArc(2, 3, 2, 3, 3.1);
    g.addArc(3, 4, 3, 4, 4.1);
    g.addArc(4, 5, 4, Graph::epsilon, 5.1);
    std::stringstream stream;
    save(stream, g);
    Graph g2 = load(stream);
    CHECK(equal(g, g2));
    CHECK(isomorphic(g, g2));

    save(fn, g);
    Graph g3 = load(fn);
    CHECK(equal(g, g3));
    CHECK(isomorphic(g, g3));
  }
}
