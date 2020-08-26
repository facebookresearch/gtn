#include "gtn/common/conversions.h"

namespace gtn {

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
