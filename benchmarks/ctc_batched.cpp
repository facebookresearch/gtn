#include <cstdlib>

#include "benchmarks/time_utils.h"
#include "gtn/gtn.h"

using namespace gtn;

std::vector<int> many_rands(int num) {
  std::vector<int> out(num);
  for (int i = 0; i < num; ++i) {
    out[i] = std::rand();
  }
  return out;
}

// For emissions generation
std::vector<float> emissions(int num) {
  std::vector<float> out(num);
  auto manyRands = many_rands(num);
  for (int i = 0; i < num; ++i) {
    auto uni = manyRands[i] / static_cast<float>(RAND_MAX);
    out[i] = uni * 10 - 5;
  }
  return out;
}

std::vector<int> rand_target(int U, int N) {
  std::vector<int> target;
  auto rands = many_rands(U);
  for (int u = 0; u < U; u++) {
    // Random integer between [1, N-1]
    auto t = rands[u] % (N - 1) + 1;
    target.push_back(t);
  }
  return target;
}

Graph ctc_graph(int U, int N, std::vector<int> target) {
  int blank = 0;
  int L = 2 * U + 1;
  Graph ctc;
  for (int l = 0; l < L; l++) {
    int idx = (l - 1) / 2;
    ctc.addNode(l == 0, l == L - 1 || l == L - 2);
    int label = l % 2 ? target[idx] : blank;
    ctc.addArc(l, l, label);
    if (l > 0) {
      ctc.addArc(l - 1, l, label);
    }
    if (l % 2 && l > 1 && label != target[idx - 1]) {
      ctc.addArc(l - 2, l, label);
    }
  }
  return ctc;
}

Graph emission_graph(int T, int N, std::vector<float> scores) {
  assert(scores.size() == T * N);
  Graph emissions;
  emissions.addNode(true);
  for (int t = 1; t <= T; t++) {
    emissions.addNode(false, t == T);
    for (int i = 0; i < N; i++) {
      emissions.addArc(t - 1, t, i, i, scores[t * N + i]);
    }
  }
  return emissions;
}

int main() {
  /* Various CTC benchmarks. */

  const int T = 1000; // input frames
  const int U = 100; // output tokens
  const int N = 28; // size of alphabet
  const int B = 8;

  // Pre-compute rand targets to avoid contention with std::rand.
  // Pre-compute emissions scores
  std::vector<std::vector<int>> targets;
  std::vector<std::vector<float>> emissionsScores;
  for (int64_t b = 0; b < B; ++b) {
    targets.push_back(std::move(rand_target(U, N)));
    emissionsScores.push_back(std::move(emissions(T * N)));
  }

  auto ctc_batched = [T, U, N, B, &targets, &emissionsScores]() {
    // Loss graphs
    std::vector<Graph> vec(B);

#pragma omp parallel for num_threads(B)
    for (int64_t b = 0; b < B; ++b) {
      auto ctc = std::move(ctc_graph(U, N, targets[b]));
      auto emissions = std::move(emission_graph(T, N, emissionsScores[b]));

      vec[b] = std::move(subtract(
          forwardScore(emissions), forwardScore(compose(ctc, emissions))));
    }

#pragma omp parallel for num_threads(B)
    for (int64_t b = 0; b < B; ++b) {
      backward(std::move(vec[b]));
      vec[b] = Graph{}; // parallelize destruction
    }
  };

  TIME(ctc_batched);
}
