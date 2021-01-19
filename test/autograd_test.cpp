/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define CATCH_CONFIG_MAIN

#include <functional>

#include "catch.hpp"

#include "gtn/autograd.h"
#include "gtn/creations.h"
#include "gtn/functions.h"
#include "gtn/graph.h"
#include "gtn/rand.h"
#include "gtn/utils.h"

using namespace gtn;

bool isclose(float a, float b, float relTol = 1e-5, float absTol = 1e-3) {
  return std::abs(a - b) <=
      std::max(relTol * std::max(std::abs(a), std::abs(b)), absTol);
}

using ForwardFunction = std::function<Graph(Graph)>;

// Currently assumes that func returns a scalar graph.
bool numericalGradCheck(
    const ForwardFunction& func,
    Graph& input,
    float epsilon,
    float relTol) {
  // Numerical gradient check.
  bool gradPass = true;
  for (auto a = 0; a < input.numArcs(); ++a) {
    auto weight = input.weight(a);
    input.setWeight(a, weight + epsilon);
    auto high = func(input).item();
    input.setWeight(a, weight - epsilon);
    auto low = func(input).item();
    auto numgrad = (high - low) / (2 * epsilon);
    gradPass &= isclose(input.grad().weight(a), numgrad, relTol);
    input.setWeight(a, weight);
  }
  return gradPass;
}

TEST_CASE("Test Autograd", "[autograd]") {
  // The graph is not retained by default
  auto g1 = scalarGraph(3.0);
  auto g2 = scalarGraph(3.0);

  auto result = add(g1, g2);
  backward(result);
  CHECK(result.inputs().empty());
  // Cannot backward twice when graph is cleared.
  CHECK_THROWS(backward(result));

  // Check the graph is retained
  g1.zeroGrad();
  g2.zeroGrad();
  result = add(g1, g2);
  backward(result, true);
  CHECK(result.inputs().size() == 2);
  result.zeroGrad();
  g1.zeroGrad();
  g2.zeroGrad();
  backward(result, true);
  CHECK(g1.grad().item() == 1.0);
  CHECK(g2.grad().item() == 1.0);

  // Check that provided input gradients are used.
  g1.zeroGrad();
  g2.zeroGrad();
  result = add(g1, g2);
  Graph deltas;
  deltas.addNode(true);
  deltas.addNode(false, true);
  deltas.addArc(0, 1, 0, 0, 7.0);
  backward(result, deltas);
  CHECK(g1.grad().item() == 7.0);
  CHECK(g2.grad().item() == 7.0);
}

TEST_CASE("Test Scalar Ops Grad", "[functions.scalar (grad)]") {
  auto g1 = scalarGraph(3.0);

  auto result = negate(g1);
  backward(result);
  CHECK(g1.grad().item() == -1.0f);

  g1.zeroGrad();

  auto g2 = scalarGraph(4.0);

  result = add(g1, g2);
  backward(result);
  CHECK(g1.grad().item() == 1.0f);
  CHECK(g2.grad().item() == 1.0f);

  g1.zeroGrad();
  g2.zeroGrad();

  result = subtract(g1, g2);
  backward(result);
  CHECK(g1.grad().item() == 1.0f);
  CHECK(g2.grad().item() == -1.0f);
  g1.zeroGrad();
  g2.zeroGrad();

  result = add(add(g1, g2), g1);
  backward(result);
  CHECK(g1.grad().item() == 2.0f);
  CHECK(g2.grad().item() == 1.0f);
  g1.zeroGrad();

  auto g2nograd = scalarGraph(4.0, /* calcGrad = */ false);

  result = add(g1, g2nograd);
  backward(result);
  CHECK(g1.grad().item() == 1.0f);
  CHECK_THROWS(g2nograd.grad());
}

TEST_CASE("Test Clone/Project Grad", "[functions.clone (grad)]") {
  auto g1 = scalarGraph(3.0);
  auto g2 = scalarGraph(4.0);

  auto cloned = clone(g1);

  auto result = add(g1, g2);
  backward(result);

  // Cloned wasn't used in the computation
  CHECK_THROWS(cloned.grad());

  // Cloned was used in the computation
  g1.zeroGrad();
  g2.zeroGrad();
  result = add(cloned, g2);
  backward(result);
  CHECK(equal(cloned.grad(), g1.grad()));
}

TEST_CASE("Test Compose Grad", "[functions.compose (grad)]") {
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

  Graph composed = compose(first, second);
  backward(composed);

  std::vector<float> gradsFirst = {1, 0, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0};
  std::vector<float> gradsSecond = {1, 2, 3, 2};
  for (int i = 0; i < gradsFirst.size(); i++) {
    CHECK(gradsFirst[i] == first.grad().weight(i));
  }
  for (int i = 0; i < gradsSecond.size(); i++) {
    CHECK(gradsSecond[i] == second.grad().weight(i));
  }
}

TEST_CASE("Test Compose Epsilon Grad", "[functions.compose_epsilon (grad)]") {
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

  Graph composed = compose(first, second);

  backward(composed);

  auto& grad1 = first.grad();
  auto& grad2 = second.grad();
  std::vector<float> expectedFirstGrad = {3, 3, 3, 5};
  std::vector<float> expectedSecondGrad = {1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 2};
  for (size_t i = 0; i < grad1.numArcs(); ++i) {
    CHECK(grad1.weights()[i] == expectedFirstGrad[i]);
  }
  for (size_t i = 0; i < grad2.numArcs(); ++i) {
    CHECK(grad2.weights()[i] == expectedSecondGrad[i]);
  }
}

TEST_CASE("Test Grad Available", "[functions.isGradAvailable (grad)]") {
  {
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
    CHECK(!g.isGradAvailable());
    backward(forwardScore(g));
    CHECK(g.isGradAvailable());
  }
}

TEST_CASE("Test forwardScore Grad", "[functions.forwardScore (grad)]") {
  {
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
    backward(forwardScore(g));
    CHECK(numericalGradCheck(forwardScore, g, 1e-3, 1e-3));
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
    backward(forwardScore(g));
    CHECK(numericalGradCheck(forwardScore, g, 1e-3, 1e-3));

    double denom = 1 / (std::exp(-3) + std::exp(1) + std::exp(2));
    auto grad = g.grad();
    CHECK(grad.weight(0) == Approx(denom * std::exp(-3)));
    CHECK(grad.weight(1) == Approx(denom * std::exp(1)));
    CHECK(grad.weight(2) == Approx(denom * (std::exp(-3) + std::exp(2))));
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
    backward(forwardScore(g));
    CHECK(numericalGradCheck(forwardScore, g, 1e-3, 1e-3));

    double denom = 1 / (2 * std::exp(2) + std::exp(4));
    auto& grad = g.grad();
    CHECK(grad.weight(0) == Approx(denom * (std::exp(2) + std::exp(4))));
    CHECK(grad.weight(1) == Approx(denom * std::exp(2)));
    CHECK(grad.weight(2) == Approx(denom * std::exp(4)));
  }

  {
    // Handle case where some arcs don't lead to accepting states
    Graph g;
    g.addNode(true);
    g.addNode(false, false);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, 2);
    g.addArc(0, 2, 0, 0, 2);
    backward(forwardScore(g));
    CHECK(numericalGradCheck(forwardScore, g, 1e-3, 1e-3));
    auto& grad = g.grad();
    CHECK(grad.weight(0) == Approx(0.0));
    CHECK(grad.weight(1) == Approx(1.0));
  }

  const float inf = std::numeric_limits<float>::infinity();
  {
    // Handles negative infinity
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, -inf);
    g.addArc(0, 1, 1, 1, -inf);
    backward(forwardScore(g));

    auto& grad = g.grad();
    CHECK(std::isnan(grad.weight(0)));
    CHECK(std::isnan(grad.weight(1)));

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, -inf);
    g2.addArc(0, 1, 1, 1, 1.0);
    backward(forwardScore(g2));

    auto& grad2 = g2.grad();
    CHECK(grad2.weight(0) == Approx(0.0));
    CHECK(grad2.weight(1) == Approx(1.0));
  }

  {
    // Handles infinity
    Graph g;
    g.addNode(true);
    g.addNode(false, true);
    g.addArc(0, 1, 0, 0, inf);
    g.addArc(0, 1, 1, 1, inf);
    backward(forwardScore(g));
    auto& grad = g.grad();
    CHECK(std::isnan(grad.weight(0)));
    CHECK(std::isnan(grad.weight(1)));

    Graph g2;
    g2.addNode(true);
    g2.addNode(false, true);
    g2.addArc(0, 1, 0, 0, inf);
    g2.addArc(0, 1, 1, 1, 1.0);
    backward(forwardScore(g2));
    auto& grad2 = g2.grad();
    CHECK(std::isnan(grad2.weight(0)));
    CHECK(std::isnan(grad2.weight(1)));
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
    Graph g = loadTxt(in);
    backward(forwardScore(g));
    CHECK(numericalGradCheck(forwardScore, g, 1e-3, 1e-3));
  }
}

TEST_CASE("Test viterbiScore Grad", "[functions.viterbiScore (grad)]") {
  auto gradsToVec = [](Graph g) {
    std::vector<float> grads;
    for (auto a = 0; a < g.numArcs(); ++a) {
      grads.push_back(g.grad().weight(a));
    }
    return grads;
  };

  {
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
    backward(viterbiScore(g));
    std::vector<float> expected = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    CHECK(gradsToVec(g) == expected);
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
    backward(viterbiScore(g));
    std::vector<float> expected = {0.0, 0.0, 1.0};
    CHECK(gradsToVec(g) == expected);
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
    backward(viterbiScore(g));
    std::vector<float> expected = {1.0, 0.0, 1.0};
    CHECK(gradsToVec(g) == expected);
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
    Graph g = loadTxt(in);
    backward(viterbiScore(g));
    // two possible paths with same viterbi score
    std::vector<float> expected1 = {1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    std::vector<float> expected2 = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    CHECK(((gradsToVec(g) == expected1) || (gradsToVec(g) == expected2)));
  }
}

TEST_CASE("Test viterbiPath Grad", "[functions.viterbiPath (grad)]") {
  auto gradsToVec = [](Graph g) {
    std::vector<float> grads;
    for (auto a = 0; a < g.numArcs(); ++a) {
      grads.push_back(g.grad().weight(a));
    }
    return grads;
  };

  {
    std::stringstream in(
        "0 1\n"
        "3 4\n"
        "0 1 0 0 2\n"
        "0 2 1 1 1\n"
        "1 2 0 0 2\n"
        "2 3 0 0 1\n"
        "2 3 1 1 3\n"
        "1 4 0 0 2\n"
        "2 4 1 1 3\n"
        "3 4 0 0 2\n");
    Graph g = loadTxt(in);
    backward(viterbiPath(g));
    std::vector<float> expected = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    CHECK(gradsToVec(g) == expected);
    g.zeroGrad();

    auto forwardFn = [](Graph g) {
      auto paths = {viterbiPath(g), viterbiPath(g), viterbiPath(g)};
      return forwardScore(union_(paths));
    };
    backward(forwardFn(g));

    CHECK(numericalGradCheck(forwardFn, g, 1e-2, 1e-5));
  }
}

TEST_CASE("Test Sample Grad", "[rand.sample (grad)]") {
  Graph g;
  g.addNode(true);
  g.addNode();
  g.addNode(false, true);
  g.addArc(0, 0, 0);
  g.addArc(0, 1, 1);
  g.addArc(1, 0, 2);
  g.addArc(1, 2, 3);

  for (int i = 0; i < 5; i++) {
    g.zeroGrad();
    auto path = sample(g);
    // One for each arc in the original graph
    std::vector<float> grads = {0.0, 0.0, 0.0, 0.0};
    for (auto a = 0; a < path.numArcs(); ++a) {
      grads[path.label(a)]++;
    }
    backward(path);
    for (int i = 0; i < grads.size(); i++) {
      CHECK(grads[i] == g.grad().weight(i));
    }
  }
}

TEST_CASE("Test Sum Grad", "[functions.union_ (grad)]") {
  Graph g1;
  g1.addNode(true);
  g1.addNode();
  g1.addNode(false, true);
  g1.addArc(0, 1, 0);
  g1.addArc(1, 2, 1);

  // Works with a no gradient graph
  Graph g2(false);
  g2.addNode(true);
  g2.addNode();
  g2.addNode(false, true);
  g2.addArc(0, 1, 0);
  g2.addArc(1, 2, 1);

  Graph g3;
  g3.addNode(true);
  g3.addNode();
  g3.addNode(false, true);
  g3.addArc(0, 1, 0);
  g3.addArc(1, 2, 1);

  backward(forwardScore(union_({g1, g2, g3})));

  auto forwardFn1 = [g2, g3](Graph g) {
    return forwardScore(union_({g, g2, g3}));
  };
  CHECK(numericalGradCheck(forwardFn1, g1, 1e-4, 1e-3));

  auto forwardFn2 = [g1, g2](Graph g) {
    return forwardScore(union_({g1, g2, g}));
  };
  CHECK(numericalGradCheck(forwardFn2, g3, 1e-4, 1e-3));

  CHECK_THROWS(g2.grad());
}

TEST_CASE("Test Concat Grad", "[functions.concat (grad)]") {
  Graph g1;
  g1.addNode(true);
  g1.addNode();
  g1.addNode(false, true);
  g1.addArc(0, 1, 0);
  g1.addArc(1, 2, 1);

  // Works with a no gradient graph
  Graph g2(false);
  g2.addNode(true);
  g2.addNode();
  g2.addNode(false, true);
  g2.addArc(0, 1, 0);
  g2.addArc(1, 2, 1);

  Graph g3;
  g3.addNode(true);
  g3.addNode();
  g3.addNode(false, true);
  g3.addArc(0, 1, 0);
  g3.addArc(1, 2, 1);

  backward(forwardScore(concat({g1, g2, g3})));

  auto forwardFn1 = [g2, g3](Graph g) {
    return forwardScore(concat({g, g2, g3}));
  };
  CHECK(numericalGradCheck(forwardFn1, g1, 1e-4, 1e-3));

  auto forwardFn2 = [g1, g2](Graph g) {
    return forwardScore(concat({g1, g2, g}));
  };
  CHECK(numericalGradCheck(forwardFn2, g3, 1e-4, 1e-3));

  CHECK_THROWS(g2.grad());
}

TEST_CASE("Test Closure Grad", "[functions.closure (grad)]") {
  Graph g1;
  g1.addNode(true);
  g1.addNode(false, true);
  g1.addArc(0, 1, 0, 0, 1.3);
  g1.addArc(1, 1, 1, 1, 2.1);

  Graph g2;
  g2.addNode(true);
  g2.addNode();
  g2.addNode();
  g2.addNode();
  g2.addNode(false, true);
  g2.addArc(0, 1, 0);
  g2.addArc(0, 1, 1);
  g2.addArc(1, 2, 0);
  g2.addArc(1, 2, 1);
  g2.addArc(2, 3, 0);
  g2.addArc(2, 3, 1);
  g2.addArc(3, 4, 0);
  g2.addArc(3, 4, 1);

  backward(forwardScore(compose(closure(g1), g2)));

  auto forwardFn = [g2](Graph g) {
    return forwardScore(compose(closure(g), g2));
  };
  CHECK(numericalGradCheck(forwardFn, g1, 1e-3, 1e-3));
}
