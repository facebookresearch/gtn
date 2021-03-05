/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdlib>

#include "gtn/gtn.h"

using namespace gtn;

/*
 * Use GTN to count n-grams in a string.
 *
 * Given an input sequence, say x = ['a', 'a', 'b', 'a', 'a'] and an alphabet
 * {'a', 'b'}, we want to count the number of n-grams. For example, if n = 2,
 * the count of ('a', 'a') bigrams in x is 2, the count of ('b', 'a') bigrams
 * is 1 and the count of ('a', 'b') bigrams is 1.
 */

Graph makeNgramGraph(const int n, const int numTokens) {
  // Makes a graph that can be used to count the number of ocurrences of a
  // given n-gram in an input.
  Graph ngramCounter = linearGraph(n, numTokens);

  // Add epsilon transitions at the beginning and end.
  for (int i = 0; i < numTokens; ++i) {
    ngramCounter.addArc(0, 0, i, gtn::epsilon);
    ngramCounter.addArc(n, n, i, gtn::epsilon);
  }
  return ngramCounter;
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
  // Toy example
  {
    int numTokens = 2; // size of alphabet
    int n = 2; // n in n-gram

    // Make an input
    auto input = makeChainGraph({0, 1, 0, 1});

    // Make an n-gram
    auto ngram = makeChainGraph({0, 1});

    // Make an n-gram counting graph
    auto ngramCounter = makeNgramGraph(n, numTokens);

    // Compose and score to get counts
    auto score = forwardScore(compose(input, compose(ngramCounter, ngram)));
    // Extract score from graph which is in log space
    int count = std::exp(score.item());
    assert(count == 2);
  }

  // Larger example
  {
    int numTokens = 28; // size of alphabet
    int n = 3; // n in n-gram
    auto ngramCounter = makeNgramGraph(n, numTokens);
    auto input = makeChainGraph({0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1});
    std::vector<std::vector<int>> ngrams = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
    for (auto& ngram : ngrams) {
      auto ngramGraph = makeChainGraph(ngram);
      auto score =
          forwardScore(compose(input, compose(ngramCounter, ngramGraph)));
      auto count = std::exp(score.item());
      assert(count == 3);
    }
  }
}
