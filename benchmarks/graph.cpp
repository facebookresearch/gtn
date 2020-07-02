#include <queue>

#include "benchmarks/time_utils.h"
#include "gtn/gtn.h"

using namespace gtn;

void timeConstructDestruct() {
  // Hold the reference to the graphs so we don't time destruction.
  std::vector<Graph> graphs;
  auto linearConstruction = [&graphs]() {
    graphs.push_back(makeLinear(1000, 1000));
  };
  TIME(linearConstruction);

  auto linearDestruction = [&graphs]() { graphs.pop_back(); };
  TIME(linearDestruction);
}

void timeCopy() {
  auto graph = makeLinear(1000, 1000);
  auto copy = [&graph]() { auto copied = Graph::deepCopy(graph); };
  TIME(copy);
}

void timeTraversal() {
  auto graph = makeLinear(100000, 100);

  // A simple iterative function to visit every node in a graph.
  auto traverseForward = [&graph]() {
    std::vector<bool> visited(graph.numNodes(), false);
    std::queue<int> toExplore;
    for (auto s : graph.start()) {
      toExplore.push(s);
    }
    while (!toExplore.empty()) {
      auto curr = toExplore.front();
      toExplore.pop();
      for (auto a : graph.out(curr)) {
        auto dn = graph.downNode(a);
        if (!visited[dn]) {
          visited[dn] = true;
          toExplore.push(dn);
        }
      }
    }
  };
  TIME(traverseForward);

  auto traverseBackward = [&graph]() {
    std::vector<bool> visited(graph.numNodes(), false);
    std::queue<int> toExplore;
    for (auto a : graph.accept()) {
      toExplore.push(a);
    }
    while (!toExplore.empty()) {
      auto curr = toExplore.front();
      toExplore.pop();
      for (auto a : graph.in(curr)) {
        auto un = graph.upNode(a);
        if (!visited[un]) {
          visited[un] = true;
          toExplore.push(un);
        }
      }
    }
  };
  TIME(traverseBackward);
}

int main() {
  /* Various function benchmarks. */
  timeConstructDestruct();
  timeCopy();
  timeTraversal();
}
