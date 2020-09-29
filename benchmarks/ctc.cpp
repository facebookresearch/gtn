/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <math.h>

#include "benchmarks/time_utils.h"
#include "gtn/gtn.h"

using namespace gtn;

float randScore() {
  auto uni = static_cast<float>(std::rand());
  uni /= static_cast<float>(RAND_MAX);
  return uni * 10 - 5;
}

std::vector<int> randTarget(int U, int M) {
  std::vector<int> target;
  for (int u = 0; u < U; u++) {
    // Random integer between [1, M-1]
    auto t = rand() % (M - 1) + 1;
    target.push_back(t);
  }
  return target;
}

std::vector<float> randVec(int num) {
  std::vector<float> out(num);
  for (int i = 0; i < num; ++i) {
    out[i] = randScore();
  }
  return out;
}

Graph ctcGraph(const std::vector<int>& target) {
  int blank = 0;
  int L = 2 * target.size() + 1;
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
  ctc.arcSort();
  return ctc;
}

/* Build a dense `N`-gram transition graph with an alphabet size `M`. The graph
 * will have `M^{N-1}` nodes and `M^N` arcs. Each node represents the state
 * `(x_{N-1}, ..., x_1)` and has `M` outgoing arcs which represent the `N`-gram
 * score `s(x_N, ..., x_1)`. State `(x_{N-1}, ..., x_1)` transitions to state
 * `(x_N, ..., x_2)` on label `x_N`.
 */
Graph transitionsGraph(int M, int N) {
  auto numNodes = static_cast<int>(std::pow(M, N - 1));
  Graph graph;
  for (int i = 0; i < numNodes; i++) {
    graph.addNode(true, true);
  }
  auto modVal = numNodes / M;
  for (int i = 0; i < numNodes; ++i) {
    for (int m = 0; m < M; ++m) {
      graph.addArc(i, i % modVal, m, m);
    }
  }
  graph.arcSort();
  return graph;
}

void timeCtc() {
  const int T = 1000; // input frames
  const int U = 100; // output tokens
  const int M = 28; // size of alphabet

  Graph ctc = ctcGraph(randTarget(U, M));
  Graph emissions = linearGraph(T, M);
  emissions.arcSort();
  emissions.setWeights(randVec(T * M).data());

  auto ctcLoss = [&ctc, &emissions]() {
    auto loss = subtract(
        forwardScore(emissions), forwardScore(intersect(ctc, emissions)));
    return loss;
  };
  TIME(ctcLoss);

  auto ctcGrad = [loss = ctcLoss(), &emissions, &ctc]() {
    emissions.zeroGrad();
    ctc.zeroGrad();
    backward(loss, true);
  };
  TIME(ctcGrad);
}

void timeNgramCtc() {
  const int T = 200; // input frames
  const int U = 10; // output tokens
  const int M = 30; // size of alphabet
  const int N = 2; // N-gram size

  Graph ctc = ctcGraph(randTarget(U, M));
  Graph emissions = linearGraph(T, M);
  emissions.setWeights(randVec(T * M).data());
  Graph transitions = transitionsGraph(M, N);

  auto ngramCtcLoss = [&ctc, &emissions, &transitions]() {
    auto num = forwardScore(intersect(intersect(ctc, transitions), emissions));
    auto denom = forwardScore(intersect(emissions, transitions));
    auto loss = subtract(denom, num);
    return loss;
  };
  TIME(ngramCtcLoss);

  auto ngramCtcGrad =
      [loss = ngramCtcLoss(), &emissions, &ctc, &transitions]() {
        emissions.zeroGrad();
        ctc.zeroGrad();
        transitions.zeroGrad();
        backward(loss, true);
      };
  TIME(ngramCtcGrad);
}

void timeBatchedCtc(const int B) {
  const int T = 1000; // input frames
  const int U = 100; // output tokens
  const int M = 28; // size of alphabet

  // Pre-compute rand targets to avoid contention with std::rand.
  // Pre-compute emissions scores
  std::vector<std::vector<int>> targets;
  std::vector<std::vector<float>> emissionsScores;
  for (int64_t b = 0; b < B; ++b) {
    targets.push_back(randTarget(U, M));
    emissionsScores.push_back(randVec(T * M));
  }

  auto fwd = [T, M, &targets, &emissionsScores](
                 const std::vector<int>& target,
                 const std::vector<float>& emissionsScore) {
    auto ctc = ctcGraph(target);
    auto emissions = linearGraph(T, M);
    emissions.setWeights(emissionsScore.data());
    return subtract(
        forwardScore(emissions), forwardScore(intersect(ctc, emissions)));
  };

  auto bwd = [](const Graph& g) { backward(g); };

  auto ctcBatched = [&targets, &emissionsScores, &fwd, &bwd]() {
    auto lossGraphs = parallelMap(fwd, targets, emissionsScores);
    parallelMap(bwd, lossGraphs);
  };

  TIME(ctcBatched);
}

int main(int argc, char** argv) {
  /* Various CTC benchmarks.
   * Usage:
   *   `./benchmark_ctc <batch_size (default 8)>`
   */
  timeCtc();
  timeNgramCtc();

  int B = 8; // batch size
  if (argc > 1) {
    B = std::stoi(argv[1]);
  }
  timeBatchedCtc(B);
}
