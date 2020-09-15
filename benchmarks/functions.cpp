#include "benchmarks/time_utils.h"
#include "gtn/gtn.h"

using namespace gtn;

void timeSimpleOps() {
  // time clone
  auto graph = makeLinear(1000, 100);
  auto cloneForward = [&graph]() { auto cloned = clone(graph); };
  TIME(cloneForward);

  auto cloneBackward = [&graph, out = clone(graph)]() {
    graph.zeroGrad();
    backward(out, true);
  };
  TIME(cloneBackward);

  // time closure
  auto closureForward = [&graph]() { auto closed = closure(graph); };
  TIME(closureForward);

  auto closureBackward = [&graph, out = closure(graph)]() {
    graph.zeroGrad();
    backward(out, true);
  };
  TIME(closureBackward);

  // time union_
  std::vector<Graph> graphs;
  for (int i = 0; i < 100; i++) {
    graphs.push_back(makeLinear(1000, 1));
  }

  auto unionForward = [&graphs]() { auto out = union_(graphs); };
  TIME(unionForward);

  auto unionBackward = [&graphs, out = union_(graphs)]() {
    for (auto& g : graphs) {
      g.zeroGrad();
    }
    backward(out, true);
  };
  TIME(unionBackward);

  // time concatenate
  auto concatForward = [&graphs]() { auto closed = concat(graphs); };
  TIME(concatForward);

  auto concatBackward = [&graphs, out = concat(graphs)]() {
    for (auto& g : graphs) {
      g.zeroGrad();
    }
    backward(out, true);
  };
  TIME(concatBackward);
}

void timeForward() {
  auto graph = makeLinear(1000, 1000);
  auto forwardScoreLinearForward = [&graph]() {
    auto out = forwardScore(graph);
  };
  TIME(forwardScoreLinearForward);

  auto forwardScoreLinearBackward = [&graph, out = forwardScore(graph)] {
    graph.zeroGrad();
    backward(out, true);
  };
  TIME(forwardScoreLinearBackward);

  graph = makeRandomDAG(2000, 20000);
  auto forwardScoreRandDAGForward = [&graph]() {
    auto out = forwardScore(graph);
  };
  TIME(forwardScoreRandDAGForward);

  auto forwardScoreRandDAGBackward = [&graph, out = forwardScore(graph)] {
    graph.zeroGrad();
    backward(out, true);
  };
  TIME(forwardScoreRandDAGBackward);
}

void timeCompose() {
  const int N1 = 100;
  const int N2 = 50;
  const int A1 = 20;
  const int A2 = 500;
  auto first = makeLinear(N1, A1);
  auto second = makeLinear(N2, A2);
  for (int i = 0; i < N2; i++) {
    for (int j = 0; j < A2; j++) {
      // add self loops so composition completes
      second.addArc(i, i, j);
    }
  }
  auto out = compose(first, second);
  auto composeForward = [&first, &second]() {
    auto out = compose(first, second);
  };
  TIME(composeForward);

  auto composeBackward = [&first, &second, out = compose(first, second)] {
    first.zeroGrad();
    second.zeroGrad();
    backward(out, true);
  };
  TIME(composeBackward);

  first.arcSort(true);
  second.arcSort();
  auto composeForwardSorted = [&first, &second]() {
    auto out = compose(first, second);
  };
  TIME(composeForwardSorted) {}
}

int main() {
  /* Various function benchmarks. */
  timeSimpleOps();
  timeForward();
  timeCompose();
}
