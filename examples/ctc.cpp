#include <cstdlib>
#include <sstream>

#include "gtn/gtn.h"

using namespace gtn;

float rand_score() {
  auto uni = static_cast<float>(std::rand());
  uni /= static_cast<float>(RAND_MAX);
  return uni * 10 - 5;
}

Graph ctc_graph(std::vector<int> target, int blank) {
  int L = target.size();
  int U = 2 * L + 1;
  Graph ctc;
  for (int l = 0; l < U; l++) {
    int idx = (l - 1) / 2;
    ctc.addNode(l == 0, l == U - 1 || l == U - 2);
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

int main() {
  /* An implementation of Connectionist Temporal Classification (CTC)
   * in the graph transducer framework. See e.g.
   * https://www.cs.toronto.edu/~graves/icml_2006.pdf
   */

  // Force align graph for "cat"
  int N = 28; // size of alphabet (arcs per step)
  Graph ctc = ctc_graph({1, 3, 21}, 0);

  // Emissions (recognition) graph
  int T = 10; // graph length (frames)
  Graph emissions;
  emissions.addNode(true);
  for (int t = 1; t <= T; t++) {
    emissions.addNode(false, t == T);
    for (int i = 0; i < N; i++) {
      emissions.addArc(t - 1, t, i, i, rand_score());
    }
  }

  auto denom = forward(emissions);
  auto composed_ctc = compose(ctc, emissions);
  auto num = forward(composed_ctc);
  auto loss = subtract(denom, num);
  std::cout << "Composed CTC Graph Nodes: " << composed_ctc.numNodes()
            << " Arcs: " << composed_ctc.numArcs() << std::endl;
  std::cout << "CTC Loss: " << loss.item() << std::endl;

  // Just for fun, add bi-gram transitions to CTC.
  Graph transitions;
  transitions.addNode(true);
  for (int i = 1; i <= N; i++) {
    transitions.addNode(false, true);
    transitions.addArc(0, i, i - 1); // p(i | <s>)
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      transitions.addArc(i + 1, j + 1, j); // p(j | i)
    }
  }

  auto composed_bigram_ctc = compose(compose(ctc, emissions), transitions);
  num = forward(composed_bigram_ctc);
  denom = forward(compose(emissions, transitions));
  loss = subtract(denom, num);
  std::cout << "Composed bi-gram CTC Graph Nodes: "
            << composed_bigram_ctc.numNodes()
            << " Arcs: " << composed_bigram_ctc.numArcs() << std::endl;
  std::cout << "Bi-gram CTC Loss " << loss.item() << std::endl;
}
