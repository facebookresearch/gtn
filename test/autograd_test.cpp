#define CATCH_CONFIG_MAIN

#include <functional>

#include "catch.hpp"

#include "gtn/autograd.h"
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
  for (int i = 0; i < input.numArcs(); i++) {
    auto& inArc = input.arcs()[i];
    auto weight = inArc.weight();
    inArc.setWeight(weight + epsilon);
    auto high = func(input).item();
    inArc.setWeight(weight - epsilon);
    auto low = func(input).item();
    auto numgrad = (high - low) / (2 * epsilon);
    auto& gradArc = input.grad().arcs()[i];
    gradPass &= isclose(gradArc.weight(), numgrad, relTol);
    inArc.setWeight(weight);
  }
  return gradPass;
}

TEST_CASE("Test Scalar Ops Grad", "[functions.scalar (grad)]") {
  Graph g1;
  g1.addNode(true);
  g1.addNode(false, true);
  g1.addArc(0, 1, 0, 0, 3.0);

  auto result = negate(g1);
  backward(result);
  CHECK(g1.grad().item() == -1.0f);

  g1.zeroGrad();

  Graph g2;
  g2.addNode(true);
  g2.addNode(false, true);
  g2.addArc(0, 1, 0, 0, 4.0);

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

  Graph g2nograd(false);
  g2nograd.addNode(true);
  g2nograd.addNode(false, true);
  g2nograd.addArc(0, 1, 0, 0, 4.0);

  result = add(g1, g2nograd);
  backward(result);
  CHECK(g1.grad().item() == 1.0f);
  CHECK_THROWS(g2nograd.grad());
}

TEST_CASE("Test Clone/Project Grad", "[functions.clone (grad)]") {
  Graph g1;
  g1.addNode(true);
  g1.addNode(false, true);
  g1.addArc(0, 1, 0, 0, 3.0);

  Graph g2;
  g2.addNode(true);
  g2.addNode(false, true);
  g2.addArc(0, 1, 0, 0, 4.0);

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
  CHECK(equals(cloned.grad(), g1.grad()));
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
    CHECK(gradsFirst[i] == first.grad().arcs()[i].weight());
  }
  for (int i = 0; i < gradsSecond.size(); i++) {
    CHECK(gradsSecond[i] == second.grad().arcs()[i].weight());
  }
}

TEST_CASE("Test Forward Grad", "[functions.forward (grad)]") {
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
    backward(forward(g));
    CHECK(numericalGradCheck(forward, g, 1e-3, 1e-3));
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
    backward(forward(g));
    CHECK(numericalGradCheck(forward, g, 1e-3, 1e-3));

    double denom = 1 / (std::exp(-3) + std::exp(1) + std::exp(2));
    auto grads = g.grad().arcs();
    CHECK(grads[0].weight() == Approx(denom * std::exp(-3)));
    CHECK(grads[1].weight() == Approx(denom * std::exp(1)));
    CHECK(grads[2].weight() == Approx(denom * (std::exp(-3) + std::exp(2))));
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
    backward(forward(g));
    CHECK(numericalGradCheck(forward, g, 1e-3, 1e-3));

    double denom = 1 / (2 * std::exp(2) + std::exp(4));
    auto& grads = g.grad().arcs();
    CHECK(grads[0].weight() == Approx(denom * (std::exp(2) + std::exp(4))));
    CHECK(grads[1].weight() == Approx(denom * std::exp(2)));
    CHECK(grads[2].weight() == Approx(denom * std::exp(4)));
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
    backward(forward(g));
    CHECK(numericalGradCheck(forward, g, 1e-3, 1e-3));
  }
}
/*
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
    for (auto& arc : path.arcs()) {
      grads[arc.label()]++;
    }
    backward(path);
    for (int i = 0; i < grads.size(); i++) {
      CHECK(grads[i] == g.arcs()[i].grad());
    }
  }
}*/

TEST_CASE("Test Sum Grad", "[functions.sum (grad)]") {
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

  backward(forward(sum({g1, g2, g3})));

  auto forwardFn1 = [g2, g3](Graph g) { return forward(sum({g, g2, g3})); };
  CHECK(numericalGradCheck(forwardFn1, g1, 1e-4, 1e-3));

  auto forwardFn2 = [g1, g2](Graph g) { return forward(sum({g1, g2, g})); };
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

  backward(forward(compose(closure(g1), g2)));

  auto forwardFn = [g2](Graph g) { return forward(compose(closure(g), g2)); };
  CHECK(numericalGradCheck(forwardFn, g1, 1e-3, 1e-3));
}
