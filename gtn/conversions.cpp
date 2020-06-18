#include "conversions.h"

namespace gtn {

Graph createLinearGraph(
    float* input,
    int T,
    int C,
    bool calcGrad /* = true */) {
  Graph g(calcGrad);
  g.addNode(true);
  for (int t = 1; t <= T; ++t) {
    g.addNode(false, t == T);
    auto inOffset = (t - 1) * C;
    for (int c = 0; c < C; ++c) {
      g.addArc(t - 1, t, c, c, input[inOffset + c]);
    }
  }
  return g;
}

void extractLinearGrad(Graph g, float scale, float* grad) {
  const auto& arcs = g.grad().arcs();
  for (int i = 0; i < arcs.size(); ++i) {
    grad[i] = scale * arcs[i].weight();
  }
}
} // namespace gtn
