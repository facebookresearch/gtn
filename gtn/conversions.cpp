#include "conversions.h"

namespace gtn {

Graph arrayToLinearGraph(
    float* src,
    int M,
    int N,
    bool calcGrad /* = true */) {
  Graph g(calcGrad);
  g.addNode(true);
  for (int m = 1; m <= M; ++m) {
    g.addNode(false, m == M);
    auto inOffset = (m - 1) * N;
    for (int n = 0; n < N; ++n) {
      g.addArc(m - 1, m, n, n, src[inOffset + n]);
    }
  }
  return g;
}

void linearGraphToArray(Graph g, float* dst) {
  for (int i = 0; i < g.numArcs(); ++i) {
    dst[i] = g.arcs()[i].weight();
  }
}
} // namespace gtn
