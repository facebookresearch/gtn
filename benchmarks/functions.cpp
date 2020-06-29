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
    backward(out);
  };
  TIME(cloneBackward);


  // time closure
  auto closureForward = [&graph]() { auto closed = closure(graph); };
  TIME(closureForward);

  auto closureBackward = [&graph, out = closure(graph)]() {
    graph.zeroGrad();
    backward(out);
  };
  TIME(closureBackward);

  // time sum (union)
  std::vector<Graph> graphs;
  for (int i = 0; i < 100; i++) {
    graphs.push_back(makeLinear(1000, 1));
  }

  auto sumForward = [&graphs]() { auto summed = sum(graphs); };
  TIME(sumForward);

  auto sumBackward = [&graphs, out = sum(graphs)]() {
    for (auto& g : graphs) {
      g.zeroGrad();
    }
    backward(out);
  };
  TIME(sumBackward);

  // time concatenate
  auto concatForward = [&graphs]() { auto closed = concat(graphs); };
  TIME(concatForward);

  auto concatBackward = [&graphs, out = concat(graphs)]() {
    for (auto& g : graphs) {
      g.zeroGrad();
    }
    backward(out);
  };
  TIME(concatBackward);
}

void timeForward() {
  auto graph = makeLinear(1000, 100);
  auto forwardForwardLinear = [&graph]() { auto out = forward(graph); };
  TIME(forwardForwardLinear);

  auto forwardBackwardLinear = [&graph, out = forward(graph)] {
    graph.zeroGrad();
    backward(out);
  };
  TIME(forwardBackwardLinear);

  graph = makeRandomDAG(2000, 20000);
  auto forwardForwardRandDAG = [&graph]() { auto out = forward(graph); };
  TIME(forwardForwardRandDAG);

  auto forwardBackwardRandDAG = [&graph, out = forward(graph)] {
    graph.zeroGrad();
    backward(out);
  };
  TIME(forwardBackwardRandDAG);
}

void timeCompose() {
  auto first = makeLinear(100, 30);
  auto second = makeLinear(50, 100);
  for (int i = 0; i < 50; i++) {
    for (int j = 0; j < 100; j++) {
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
    backward(out);
  };
  TIME(composeBackward);
}

int main() {
  /* Various function benchmarks. */
  timeSimpleOps();
  timeForward();
  timeCompose();
}
