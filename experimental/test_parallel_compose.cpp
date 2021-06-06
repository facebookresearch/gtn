#include <cmath>
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
  using gtn::detail::dataparallel::convertFromDataParallel;
  using gtn::detail::dataparallel::convertToDataParallel;

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
    gIn.addArc(0, 1, 0, 2, 2.1);
    gIn.addArc(0, 1, 0, 2, 3.1);
    gIn.addArc(2, 2, 1, 0, 0.1);
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

  // Weights combine properly
  {
    auto g1 = linearGraph(2, 3);
    auto g2 = linearGraph(2, 3);
    std::vector<float> w1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    std::vector<float> w2 = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    g1.setWeights(w1.data());
    g2.setWeights(w2.data());
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
    assert(isomorphic(gtn::detail::dataparallel::compose(g1, g2), expected));
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
    assert(isomorphic(gtn::detail::dataparallel::compose(g1, g2), expected));
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
    auto gOutP = gtn::detail::dataparallel::compose(g1, g2);
    assert(isomorphic(gtn::detail::dataparallel::compose(g1, g2), expected));
  }
}

void testEpsilon() {
  {
    // Simple test case for output epsilon on first graph
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 0, 0, epsilon, 1.0);
    g1.addArc(0, 1, 1, 2);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 2, 3);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addArc(0, 0, 0, epsilon, 1.0);
    expected.addArc(0, 1, 1, 3);

    assert(equal(gtn::detail::dataparallel::compose(g1, g2), expected));
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
    g2.addArc(1, 1, epsilon, 0, 2.0);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addArc(0, 1, 1, 3);
    expected.addArc(1, 1, epsilon, 0, 2.0);

    assert(equal(gtn::detail::dataparallel::compose(g1, g2), expected));
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
    g1.addArc(1, 2, symbols["b"], epsilon);
    g1.addArc(2, 3, symbols["c"], epsilon);
    g1.addArc(3, 4, symbols["d"], symbols["d"]);

    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, symbols["a"], symbols["d"]);
    g2.addArc(1, 2, epsilon, symbols["e"]);
    g2.addArc(2, 3, symbols["d"], symbols["a"]);

    Graph expected;
    expected.addNode(true);
    expected.addNode();
    expected.addNode();
    expected.addNode();
    expected.addNode(false, true);
    expected.addArc(0, 1, symbols["a"], symbols["d"]);
    expected.addArc(1, 2, symbols["b"], symbols["e"]);
    expected.addArc(2, 3, symbols["c"], epsilon);
    expected.addArc(3, 4, symbols["d"], symbols["a"]);

    assert(
        randEquivalent(gtn::detail::dataparallel::compose(g1, g2), expected));
  }

  {
    // Test multiple input/output epsilon transitions per node
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 0, 1, epsilon, 1.1);
    g1.addArc(0, 1, 2, epsilon, 2.1);
    g1.addArc(0, 1, 3, epsilon, 3.1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 3, 2.1);
    g2.addArc(0, 1, 1, 2);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addArc(0, 0, 1, epsilon, 1.1);
    expected.addArc(0, 1, 2, 3, 4.2);
    expected.addArc(0, 1, 3, 3, 5.2);

    assert(
        randEquivalent(gtn::detail::dataparallel::compose(g1, g2), expected));
  }
}

void testMoreEpsilon() {
  // A series of tests making sure we handle redundant epsilon paths correctly
  {
    Graph g1;
    g1.addNode(true, true);
    g1.addArc(0, 0, 0, epsilon);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 0, 1.0);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addArc(0, 0, 0, epsilon);
    expected.addArc(0, 1, epsilon, 0, 1.0);

    assert(
        randEquivalent(gtn::detail::dataparallel::compose(g1, g2), expected));
  }

  {
    Graph g1;
    g1.addNode(true, true);
    g1.addArc(0, 0, 0, epsilon);

    Graph g2;
    g2.addNode(true, true);
    g2.addArc(0, 0, epsilon, 0);

    Graph expected;
    expected.addNode(true, true);
    expected.addArc(0, 0, 0, epsilon);
    expected.addArc(0, 0, epsilon, 0);

    assert(
        randEquivalent(gtn::detail::dataparallel::compose(g1, g2), expected));
  }

  {
    Graph g1;
    g1.addNode(true, true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, epsilon);
    g1.addArc(1, 0, 0, epsilon);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 0, 1.0);

    Graph expected;
    expected.addNode(true);
    expected.addNode(false, true);
    expected.addArc(0, 1, epsilon, 0, 1.0);
    expected.addArc(1, 1, 0, epsilon);

    assert(
        randEquivalent(gtn::detail::dataparallel::compose(g1, g2), expected));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, epsilon);
    g1.addArc(0, 0, 0, 1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 0);
    g2.addArc(0, 1, 1, 1);

    Graph expected;
    expected.addNode(true);
    expected.addNode();
    expected.addNode(false, true);
    expected.addArc(0, 1, 0, 1);
    expected.addArc(0, 1, epsilon, 0);
    expected.addArc(1, 2, 0, epsilon);

    assert(
        randEquivalent(gtn::detail::dataparallel::compose(g1, g2), expected));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, epsilon);
    g1.addArc(0, 1, 0, 1);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 0);
    g2.addArc(0, 0, 1, 1);

    Graph expected;
    expected.addNode(true);
    expected.addNode();
    expected.addNode(false, true);
    expected.addArc(0, 1, 0, 1);
    expected.addArc(0, 1, 0, epsilon);
    expected.addArc(1, 2, epsilon, 0);

    assert(
        randEquivalent(gtn::detail::dataparallel::compose(g1, g2), expected));
  }

  {
    Graph g1;
    g1.addNode(true);
    g1.addNode(false, true);
    g1.addArc(0, 1, 0, epsilon);
    g1.addArc(0, 1, 0, 1);
    g1.addArc(0, 0, 1, 0);

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, epsilon, 0);
    g2.addArc(0, 0, 1, 0);
    g2.addArc(0, 1, 0, 1);

    Graph expected;
    expected.addNode(true);
    expected.addNode();
    expected.addNode();
    expected.addNode(false, true);
    expected.addArc(0, 1, 1, 1);
    expected.addArc(0, 1, epsilon, 0);
    expected.addArc(0, 2, 0, 0);
    expected.addArc(2, 3, epsilon, 0);
    expected.addArc(1, 3, 0, epsilon);

    assert(
        randEquivalent(gtn::detail::dataparallel::compose(g1, g2), expected));
  }
}

void testGrad() {
  Graph first;
  first.addNode(true);
  first.addNode();
  first.addNode();
  first.addNode();
  first.addNode(false, true);
  first.addArc(0, 1, 0, 0, 0);
  first.addArc(0, 1, 1, 1, 1);
  first.addArc(0, 1, 2, 2, 2);
  first.addArc(1, 2, 0, 0, 0);
  first.addArc(1, 2, 1, 1, 1);
  first.addArc(1, 2, 2, 2, 2);
  first.addArc(2, 3, 0, 0, 0);
  first.addArc(2, 3, 1, 1, 1);
  first.addArc(2, 3, 2, 2, 2);
  first.addArc(3, 4, 0, 0, 0);
  first.addArc(3, 4, 1, 1, 1);
  first.addArc(3, 4, 2, 2, 2);

  Graph second;
  second.addNode(true);
  second.addNode();
  second.addNode(false, true);
  second.addArc(0, 1, 0, 0, 3.5);
  second.addArc(1, 1, 0, 0, 2.5);
  second.addArc(1, 2, 1, 1, 1.5);
  second.addArc(2, 2, 1, 1, 4.5);

  Graph composed = gtn::detail::dataparallel::compose(first, second);
  backward(composed);

  std::vector<float> gradsFirst = {1, 0, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0};
  std::vector<float> gradsSecond = {1, 2, 3, 2};
  for (int i = 0; i < gradsFirst.size(); i++) {
    assert(gradsFirst[i] == first.grad().weight(i));
  }
  for (int i = 0; i < gradsSecond.size(); i++) {
    assert(gradsSecond[i] == second.grad().weight(i));
  }
}

void testEpsilonGrad() {
  Graph first;
  first.addNode(true);
  first.addNode(false, true);
  first.addArc(0, 0, 0, 3, 0);
  first.addArc(0, 1, 1, 4, 0);
  first.addArc(1, 1, 2, 5, 0);
  first.addArc(0, 1, 2, gtn::epsilon, 0);

  Graph second;
  second.addNode(true);
  second.addNode();
  second.addNode();
  second.addNode(false, true);
  second.addArc(0, 1, 3, 0, 0);
  second.addArc(0, 1, 3, 1, 0);
  second.addArc(0, 1, 4, 2, 0);
  second.addArc(0, 1, gtn::epsilon, 2, 0.0); // idx 3
  second.addArc(1, 2, 3, 0, 0);
  second.addArc(1, 2, 4, 1, 0);
  second.addArc(1, 2, 5, 2, 0);
  second.addArc(1, 2, gtn::epsilon, 2, 0.0); // idx 7
  second.addArc(2, 3, 4, 0, 0);
  second.addArc(2, 3, 5, 1, 0);
  second.addArc(2, 3, 5, 2, 0);
  second.addArc(2, 3, gtn::epsilon, 2, 0.0); // idx 11

  Graph expected =
      loadTxt(std::stringstream("0\n"
                                "6\n"
                                "0 1 0 0 0\n"
                                "0 1 0 1 0\n"
                                "0 3 2 -1 0\n"
                                "0 4 1 2 0\n"
                                "1 2 0 0 0\n"
                                "1 4 2 -1 0\n"
                                "1 5 1 1 0\n"
                                "2 5 2 -1 0\n"
                                "2 6 1 0 0\n"
                                "3 4 -1 2 0\n"
                                "4 5 2 2 0\n"
                                "4 5 -1 2 0\n"
                                "5 6 2 1 0\n"
                                "5 6 2 2 0\n"
                                "5 6 -1 2 0\n"));

  Graph composed = gtn::detail::dataparallel::compose(first, second);
  assert(randEquivalent(composed, expected));

  backward(composed);

  auto& grad1 = first.grad();
  auto& grad2 = second.grad();
  std::vector<float> expectedFirstGrad = {3, 3, 3, 3};
  std::vector<float> expectedSecondGrad = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  for (size_t i = 0; i < grad1.numArcs(); ++i) {
    assert(grad1.weights()[i] == expectedFirstGrad[i]);
  }
  for (size_t i = 0; i < grad2.numArcs(); ++i) {
    assert(grad2.weights()[i] == expectedSecondGrad[i]);
  }
}

Graph makeChainGraph(const std::vector<int>& input) {
  Graph chain(false);
  chain.addNode(true);
  for (auto i : input) {
    auto n = chain.addNode(false, chain.numNodes() == input.size());
    chain.addArc(n - 1, n, i);
  }
  return chain;
}

void testComposeEditDistance() {
  auto computeEditDistance = [](std::function<Graph(Graph, Graph)> compose,
                                const int numTokens,
                                const std::vector<int>& x,
                                const std::vector<int>& y) {
    // Make edits graph
    Graph edits(false);
    edits.addNode(true, true);

    for (int i = 0; i < numTokens; ++i) {
      // Add substitutions
      for (int j = 0; j < numTokens; ++j) {
        edits.addArc(0, 0, i, j, -(i != j));
      }
      // Add insertions and deletions
      edits.addArc(0, 0, i, gtn::epsilon, -1);
      edits.addArc(0, 0, gtn::epsilon, i, -1);
    }

    // Make inputs
    auto xG = makeChainGraph(x);
    auto yG = makeChainGraph(y);

    // Compose and viterbi to get distance
    auto score = viterbiScore(compose(xG, compose(edits, yG)));
    return -score.item();
  };

  // Small test case
  auto dist = computeEditDistance(
      gtn::detail::dataparallel::compose, 5, {0, 1, 0, 1}, {0, 0, 0, 1, 1});
  assert(dist == 2);

  // Larger random test cases
  const int minLength = 10;
  const int maxLength = 100;
  for (int numToks = 50; numToks < 70; numToks++) {
    // Random lengths in [minLength, maxLength)
    auto xLen = minLength + rand() % (maxLength - minLength);
    auto yLen = minLength + rand() % (maxLength - minLength);

    // Random vectors x, y with tokens in [0, numToks)
    std::vector<int> x;
    for (int i = 0; i < xLen; i++) {
      x.push_back(rand() % numToks);
    }
    std::vector<int> y;
    for (int i = 0; i < yLen; i++) {
      y.push_back(rand() % numToks);
    }

    auto dist =
        computeEditDistance(gtn::detail::dataparallel::compose, numToks, x, y);
    auto expected = computeEditDistance(compose, numToks, x, y);
    assert(dist == expected);
  }
}

void testComposeCountNgrams() {
  auto countNgrams = [](std::function<Graph(Graph, Graph)> compose,
                        const int numTokens,
                        const std::vector<int>& input,
                        const std::vector<int>& ngram) {
    // Make n-gram counting graph
    const int n = ngram.size();
    Graph ngramCounter = linearGraph(n, numTokens);
    for (int i = 0; i < numTokens; ++i) {
      ngramCounter.addArc(0, 0, i, gtn::epsilon);
      ngramCounter.addArc(n, n, i, gtn::epsilon);
    }

    // Make inputs
    auto inputG = makeChainGraph(input);
    auto ngramG = makeChainGraph(ngram);

    auto score = forwardScore(compose(inputG, compose(ngramCounter, ngramG)));
    return round(std::exp(score.item()));
  };

  // Small test
  auto counts =
      countNgrams(gtn::detail::dataparallel::compose, 2, {0, 1, 0, 1}, {0, 1});
  assert(counts == 2);

  // Larger random test cases
  const int minLength = 300;
  const int maxLength = 500;
  const int n = 3;
  const int numToks = 5;
  for (int t = 0; t < 100; t++) {
    // Random length in [minLength, maxLength)
    auto inputLen = minLength + rand() % (maxLength - minLength);

    // Random vectors input, ngram with tokens in [0, numToks)
    std::vector<int> input;
    for (int i = 0; i < inputLen; i++) {
      input.push_back(rand() % numToks);
    }
    std::vector<int> ngram;
    for (int i = 0; i < n; i++) {
      ngram.push_back(rand() % numToks);
    }

    auto count =
        countNgrams(gtn::detail::dataparallel::compose, numToks, input, ngram);
    auto expected = countNgrams(compose, numToks, input, ngram);
    assert(count == expected);
  }
}

int main() {
  testConversion();
  std::cout << "Conversion checks passed!" << std::endl;

  testNoEpsilon();
  std::cout << "No epsilon compositions passed!" << std::endl;

  testEpsilon();
  std::cout << "Epsilon compositions passed!" << std::endl;

  testMoreEpsilon();
  std::cout << "More epsilon compositions passed!" << std::endl;

  testGrad();
  std::cout << "Composition gradients passed!" << std::endl;

  testEpsilonGrad();
  std::cout << "Epsilon gradients passed!" << std::endl;

  testComposeEditDistance();
  std::cout << "Compose edit distance passed!" << std::endl;

  testComposeCountNgrams();
  std::cout << "Compose count ngrams passed!" << std::endl;
}
