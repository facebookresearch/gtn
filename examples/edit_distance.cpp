/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gtn/gtn.h"

using namespace gtn;

/*
 * Use GTN to compute the Levenshtein distance between two strings.
 */

Graph makeEditsGraph(const int numTokens) {
  Graph edits(false);
  edits.addNode(true, true);

  for (int i = 0; i < numTokens; ++i) {
    // Add substitutions
    for (int j = 0; j < numTokens; ++j) {
      edits.addArc(0, 0, i, j, -(i != j));
    }
    // Add insertions and deletions
    edits.addArc(0, 0, i, gtn::epsilon, -1);
    edits.addArc(0, 0, gtn::epsilon, i, -1);
  }
  return edits;
}

Graph makeChainGraph(const std::vector<int>& input) {
  Graph chain(false);
  chain.addNode(true);
  for (auto i : input) {
    auto n = chain.addNode(false, chain.numNodes() == input.size());
    chain.addArc(n - 1, n, i);
  }
  return chain;
}

int main() {
  {
    int numTokens = 5; // size of alphabet
    auto edits = makeEditsGraph(numTokens);

    // Make inputs
    auto x = makeChainGraph({0, 1, 0, 1});
    auto y = makeChainGraph({0, 0, 0, 1, 1});

    // Compose and viterbi to get distance
    auto score = viterbiScore(compose(x, compose(edits, y)));
    int distance = -score.item();
    assert(distance == 2);
  }

  {
    int numTokens = 5; // size of alphabet
    auto edits = makeEditsGraph(numTokens);

    // Make inputs
    auto x = makeChainGraph({0, 1, 0, 1});
    auto y = makeChainGraph({0, 1, 0, 1});

    // Compose and viterbi to get distance
    auto score = viterbiScore(compose(x, compose(edits, y)));
    int distance = -score.item();
    assert(distance == 0);
  }
}
