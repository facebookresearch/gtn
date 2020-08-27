#include "gtn/creations.h"

namespace gtn {

Graph scalarGraph(float val, bool calcGrad) {
  Graph g1(calcGrad);
  g1.addNode(true);
  g1.addNode(false, true);
  g1.addArc(0, 1, Graph::epsilon, Graph::epsilon, val);
  return g1;
}

Graph linearGraph(int M, int N, bool calcGrad /* = true */) {
  Graph g(calcGrad);
  g.addNode(true);
  for (int m = 1; m <= M; ++m) {
    g.addNode(false, m == M);
    auto inOffset = (m - 1) * N;
    for (int n = 0; n < N; ++n) {
      g.addArc(m - 1, m, n);
    }
  }
  g.markArcSorted();
  g.markArcSorted(true);
  return g;
}

} // namespace gtn
