#include <cstdlib>
#include <sstream>

#include "gtn/gtn.h"

using namespace gtn;

float rand_score() {
  auto uni = static_cast<float>(std::rand());
  uni /= static_cast<float>(RAND_MAX);
  return uni * 10 - 5;
}

int main() {
  /*
   * An implementation of the Auto Segmentation Crieterion (ASG)
   * in the graph transducer framework. See e.g.
   * https://arxiv.org/abs/1609.03193
   */

  // Force align graph for "cat"
  int L = 3;
  std::stringstream fal_str(
      "0\n"
      "3\n"
      "0 1 2\n"
      "1 1 2\n"
      "1 2 0\n"
      "2 2 0\n"
      "2 3 20\n"
      "3 3 20\n");
  Graph fal_graph = load(fal_str);
  // print(fal_graph);

  // Transition graph for the alphabet
  int N = 27; // size of alphabet (arcs per step)
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
  auto composed_fal = compose(emissions, compose(fal_graph, transitions));
  // NB: compose is associative, so we could also do:
  // auto composed_fal = compose(compose(emissions, fal_graph), transitions);
  auto composed_fcc = compose(emissions, transitions);
  auto fal = forwardScore(composed_fal);
  auto fcc = forwardScore(composed_fcc);
  auto asg = subtract(fcc, fal);
  std::cout << "Composed FAL Graph Nodes: " << composed_fal.numNodes()
            << " Arcs: " << composed_fal.numArcs() << " should be "
            << (2 * L - 1) * (T - L) + L << std::endl;
  std::cout << "Composed FCC Graph Nodes: " << composed_fcc.numNodes()
            << " Arcs: " << composed_fcc.numArcs() << " should be "
            << N * N * (T - 1) + N << std::endl;

  std::cout << "FAL: " << fal.item() << std::endl;
  std::cout << "FCC: " << fcc.item() << std::endl;
  std::cout << "ASG: " << asg.item() << std::endl;

  // Compute gradients with respect to the
  // emissions graph and the fal_graph
  backward(asg);
}
