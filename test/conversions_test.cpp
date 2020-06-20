#define CATCH_CONFIG_MAIN

#include "catch.hpp"

#include "gtn/graph.h"
#include "gtn/conversions.h"

using namespace gtn;

TEST_CASE("Test Linear Conversion", "[conversions.linear]") {
  auto rand_float = []() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  };

  int M = 5;
  int N = 10;
  std::vector<float> arr;
  for (int i = 0; i < M*N; i++) {
    arr.push_back(rand_float());
  }
  auto g = arrayToLinearGraph(arr.data(), M, N);
  CHECK(g.numNodes() == M + 1);
  CHECK(g.numArcs() == M * N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      auto idx = i * N + j;
      auto& arc = g.arcs()[idx];
      CHECK(arc.label() == j);
      CHECK(arc.weight() == arr[idx]);
    }
  }

  std::vector<float> arr_dst(M * N, 0.0);
  linearGraphToArray(g, arr_dst.data());
  CHECK(arr == arr_dst);
}


