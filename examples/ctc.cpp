/**
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
 * An implementation of Connectionist Temporal Classification (CTC)
 * in the GTN framework. See e.g.
 * https://www.cs.toronto.edu/~graves/icml_2006.pdf
 */

Graph createCtcTargetGraph(const std::vector<int>& target, int blank) {
  int L = target.size();
  int U = 2 * L + 1; // # c # a # t #
  Graph ctc;
  for (int l = 0; l < U; l++) {
    int idx = (l - 1) / 2;
    ctc.addNode(l == 0, l == U - 1 || l == U - 2);
    int label = l % 2 ? target[idx] : blank;
    ctc.addArc(l, l, label); // current label can repeat itself
    if (l > 0) {
      // transition from blank to label or vice-versa
      ctc.addArc(l - 1, l, label);
    }
    if (l % 2 && l > 1 && label != target[idx - 1]) {
      // transition from previous target label to current label provided it
      // is not a repeat label
      ctc.addArc(l - 2, l, label);
    }
  }
  return ctc;
}

int main() {
  // We consider an example where input sequence has length 5 and output
  // sequence is ['c', 'a', 't']. Output alphabet contains the letters a-z, a
  // space (_), and a blank token (#).

  int N = 28; // size of alphabet (arcs per step)
  int T = 5; // length of input
  std::vector<int> output = {3, 1, 20}; // corresponds to 'c', 'a', 't'
  Graph ctc = createCtcTargetGraph(output, 0 /* blank idx */); // https://git.io/JUKAZ

  // Emission graph
  Graph emissions = linearGraph(T, N);
  // Set the weights of the emission graph appropriately. We assume that the
  // weights are normalized with `logsoftmax` for each timestep.
  // emissions.setWeights(...);

  auto ctcAlignments = compose(ctc, emissions); // https://git.io/JUKA1
  auto ctcLoss = negate(forwardScore(ctcAlignments));
  std::cout << "CTC Alignments Graph Nodes: " << ctcAlignments.numNodes()
            << " Arcs: " << ctcAlignments.numArcs() << std::endl;
  std::cout << "CTC Loss: " << ctcLoss.item() << std::endl;

  // Compute gradients with respect to the emissions graph
  backward(ctcLoss); // emissions.grad() will be computed.
}
