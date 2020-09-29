/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define CATCH_CONFIG_MAIN

#include <vector>

#include "catch.hpp"

#include "gtn/autograd.h"
#include "gtn/creations.h"
#include "gtn/functions.h"
#include "gtn/graph.h"
#include "gtn/parallel.h"
#include "gtn/utils.h"

using namespace gtn;

TEST_CASE("Test ParallelMap One Arg", "[parallel.parallelmap.onearg]") {
  const int B = 4;

  std::vector<Graph> inputs;
  for (size_t i = 0; i < B; ++i) {
    inputs.push_back(scalarGraph(static_cast<float>(i)));
  }

  auto outputs = parallelMap(negate, inputs);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    expectedOutputs.push_back(negate(inputs[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE("Test ParallelMap Two Args", "[parallel.parallelmap.twoarg]") {
  const int B = 4;

  std::vector<Graph> inputs1;
  std::vector<Graph> inputs2;
  for (size_t i = 0; i < B; ++i) {
    inputs1.push_back(scalarGraph(static_cast<float>(i)));
    inputs2.push_back(scalarGraph(static_cast<float>(2 * i)));
  }

  auto outputs = parallelMap(add, inputs1, inputs2);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    expectedOutputs.push_back(add(inputs1[i], inputs2[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE("Test ParallelMap Broadcast", "[parallel.parallelmap.broadcast]") {
  const int B = 4;

  std::vector<Graph> inputs1;
  std::vector<Graph> inputs2 = {scalarGraph(10.0)};
  for (size_t i = 0; i < B; ++i) {
    inputs1.push_back(scalarGraph(static_cast<float>(i)));
  }

  auto outputs = parallelMap(add, inputs1, inputs2);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    // inputs2[0] should be broadcast
    expectedOutputs.push_back(add(inputs1[i], inputs2[0]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE("Test ParallelMap Lambda", "[parallel.parallelmap.lambda]") {
  auto function = [](const Graph& g1, const Graph& g2, const Graph& g3) {
    return subtract(add(g1, g2), g3);
  };

  const int B = 4;

  std::vector<Graph> inputs1;
  std::vector<Graph> inputs2;
  std::vector<Graph> inputs3;
  for (size_t i = 0; i < B; ++i) {
    inputs1.push_back(scalarGraph(static_cast<float>(i)));
    inputs2.push_back(scalarGraph(static_cast<float>(2 * i)));
    inputs3.push_back(scalarGraph(static_cast<float>(3 * i)));
  }

  auto outputs = parallelMap(function, inputs1, inputs2, inputs3);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    // inputs2[0] should be broadcast
    expectedOutputs.push_back(function(inputs1[i], inputs2[i], inputs3[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE(
    "Test ParallelMap Vector Input",
    "[parallel.parallelmap.vector_input]") {
  const int B = 4;

  std::vector<std::vector<Graph>> inputs(B);
  for (size_t i = 0; i < B; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      inputs[i].push_back(scalarGraph(static_cast<float>(i * j)));
    }
  }

  auto outputs = parallelMap(union_, inputs);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    expectedOutputs.push_back(union_(inputs[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE(
    "Test ParallelMap Vector Input 2",
    "[parallel.parallelmap.vector_input2]") {
  const int B = 4;

  std::vector<std::vector<Graph>> inputs(B);
  for (size_t i = 0; i < B; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      inputs[i].push_back(scalarGraph(static_cast<float>(i * j)));
    }
  }

  auto outputs = parallelMap(
      static_cast<Graph (*)(const std::vector<Graph>&)>(&concat), inputs);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    expectedOutputs.push_back(concat(inputs[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE(
    "Test ParallelMap Other Typed Lambda",
    "[parallel.parallelmap.other_lambda]") {
  const int B = 4;

  auto function = [](int T, int M, std::vector<float> emissionsScore) -> Graph {
    Graph g = linearGraph(T, M);
    g.setWeights(emissionsScore.data());
    return g;
  };

  int T = 2, M = 4;

  std::vector<std::vector<float>> inputs(B);
  for (size_t i = 0; i < B; ++i) {
    for (size_t j = 0; j < T * M; ++j) {
      inputs[i].push_back(static_cast<float>(i * j));
    }
  }

  std::vector<int> t({T});
  std::vector<int> m({M});
  auto outputs = parallelMap(function, t, m, inputs);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    expectedOutputs.push_back(function(T, M, inputs[i]));
  }

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(outputs[i], expectedOutputs[i]));
  }
}

TEST_CASE("Test ParallelMap Backward", "[parallel.parallelmap.backward]") {
  const int B = 4;

  std::vector<Graph> inputs1;
  std::vector<Graph> inputs2;
  std::vector<Graph> inputs1Dup;
  std::vector<Graph> inputs2Dup;
  for (size_t i = 0; i < B; ++i) {
    inputs1.push_back(scalarGraph(static_cast<float>(i)));
    inputs2.push_back(scalarGraph(static_cast<float>(2 * i)));
    inputs1Dup.push_back(scalarGraph(static_cast<float>(i)));
    inputs2Dup.push_back(scalarGraph(static_cast<float>(2 * i)));
  }

  auto outputs = parallelMap(add, inputs1, inputs2);

  std::vector<Graph> expectedOutputs;
  for (size_t i = 0; i < B; ++i) {
    auto out = add(inputs1Dup[i], inputs2Dup[i]);
    backward(out);
    expectedOutputs.push_back(out);
  }

  std::vector<bool> retainGraph({false});
  // This cast is needed because backward isn't a complete type before
  // overload resolution
  parallelMap(
      static_cast<void (*)(Graph, bool)>(backward), outputs, retainGraph);

  for (size_t i = 0; i < B; ++i) {
    CHECK(equal(inputs1[i].grad(), inputs1Dup[i].grad()));
    CHECK(equal(inputs2[i].grad(), inputs2Dup[i].grad()));
  }
}

TEST_CASE("Test ParallelMap Throws", "[parallel.parallelmap.onearg]") {
  const int B = 4;

  std::vector<Graph> inputs;
  for (size_t i = 0; i < B; ++i) {
    inputs.push_back(linearGraph(2, 1));
  }

  // Throws - inputs contains graph with more than one arc
  REQUIRE_THROWS_MATCHES(
      parallelMap(negate, inputs),
      std::logic_error,
      Catch::Message("[gtn::negate] input must have only one arc"));
}
