#include "benchmarks/time_utils.h"
#include "gtn/gtn.h"

using namespace gtn;

float rand_score() {
  auto uni = static_cast<float>(std::rand());
  uni /= static_cast<float>(RAND_MAX);
  return uni * 10 - 5;
}

std::vector<int> rand_target(int U, int N) {
  std::vector<int> target;
  for (int u = 0; u < U; u++) {
    // Random integer between [1, N-1]
    auto t = rand() % (N - 1) + 1;
    target.push_back(t);
  }
  return target;
}

Graph ctc_graph(int U, int N) {
  int blank = 0;
  std::vector<int> target = rand_target(U, N);
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

Graph emission_graph(int T, int N) {
  Graph emissions;
  emissions.addNode(true);
  for (int t = 1; t <= T; t++) {
    emissions.addNode(false, t == T);
    for (int i = 0; i < N; i++) {
      emissions.addArc(t - 1, t, i, i, rand_score());
    }
  }
  return emissions;
}

int main() {
  /* Various CTC benchmarks. */

  int T = 1000; // input frames
  int U = 100; // output tokens
  int N = 28; // size of alphabet

  Graph ctc = ctc_graph(U, N);
  Graph emissions = emission_graph(T, N);

  auto ctc_loss = [&ctc, &emissions]() {
    // Loss
    auto loss = subtract(forward(emissions), forward(compose(ctc, emissions)));
    // Gradients
    backward(loss);
  };

  TIME(ctc_loss);
}
