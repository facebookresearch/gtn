#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include <vector>

#include "gtn/common/conversions.h"
#include "gtn/graph.h"

using namespace gtn;

TEST_CASE("Test Linear Conversion", "[conversions.createLinear]") {
  auto rand_float = []() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  };

  int M = 5;
  int N = 10;
  std::vector<float> arr;
  for (int i = 0; i < M * N; i++) {
    arr.push_back(rand_float());
  }
  auto g = linearGraph(M, N);
  g.setWeights(arr.data());
  CHECK(g.numNodes() == M + 1);
  CHECK(g.numArcs() == M * N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      auto idx = i * N + j;
      CHECK(g.label(idx) == j);
      CHECK(g.weight(idx) == arr[idx]);
    }
  }

  CHECK(arr == std::vector<float>(g.weights(), g.weights() + g.numArcs()));
}
