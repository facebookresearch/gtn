/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <sstream>

#include "gtn/gtn.h"

using namespace gtn;

/*
 * An implementation of the Auto Segmentation Criterion (ASG)
 * in the graph transducer framework. See e.g.
 * https://arxiv.org/abs/1609.03193
 */

int main() {
  // We consider an example where input sequence has length 5 and output
  // sequence is ['c', 'a', 't']. Output alphabet contains the letters a-z and a
  // space (_)

  int N = 27; // size of alphabet (arcs per step)
  int T = 5; // length of input
  int L = 3; // length of target
  std::vector<int> output = {2, 0, 19}; // corresponds to 'c', 'a', 't'

  // Build emissions(recognition) graph
  Graph emissions = linearGraph(T, N);
  // emissions.setWeights(...); // set emisssion weights appropriately

  // Build transition graph
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
  // transitions.setWeights(...); // set transition weights appropriately

  // Build force align graph (all the acceptable sequences for "cat")
  Graph fal; // https://bit.ly/36to2xk
  fal.addNode(true);
  for (int l = 1; l <= L; l++) {
    int label = output[l - 1];
    fal.addNode(false, l == L);
    fal.addArc(l, l, label);
    fal.addArc(l - 1, l, label);
  }

  Graph falAlignments = compose(emissions, compose(fal, transitions)); // https://git.io/JUF9n
  // NB: compose is associative, so we could also do:
  // Graph falAlignments = compose(compose(emissions, fal), transitions);
  Graph falScore = forwardScore(falAlignments);

  // Build full connect graph (all the acceptable sequences for "cat")
  Graph fccAlignments = compose(emissions, transitions);
  Graph fccScore = forwardScore(fccAlignments);

  Graph asgScore = subtract(fccScore, falScore);
  std::cout << "FAL Alignments Graph Nodes: " << falAlignments.numNodes()
            << " Arcs: " << falAlignments.numArcs() << " (should be "
            << (2 * L - 1) * (T - L) + L << ")" << std::endl;
  std::cout << "FCC Alignments Graph Nodes: " << fccAlignments.numNodes()
            << " Arcs: " << fccAlignments.numArcs() << " (should be "
            << N * N * (T - 1) + N << ")" << std::endl;

  std::cout << "FAL Score: " << falScore.item() << std::endl;
  std::cout << "FCC Score: " << fccScore.item() << std::endl;
  std::cout << "ASG Score: " << asgScore.item() << std::endl;

  // Compute gradients with respect to emissions and transitions graphs
  backward(asgScore); // emissions.grad() and transitions.grad() will be computed.
}
