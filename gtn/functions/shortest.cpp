#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include "gtn/functions/shortest.h"

namespace gtn {
namespace detail {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline float logadd(float a, float b) {
  if (a == kNegInf) {
    return b;
  }
  if (b == kNegInf) {
    return a;
  }
  return std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
}

void shortestDistanceGrad(
    Graph& g,
    float output,
    const Graph& deltas,
    const std::vector<float>& nodeScores,
    const std::vector<float>& maxScoresCache,
    const std::vector<int>& maxArcIdxCache,
    bool tropical) {
  std::queue<int> computed;
  std::vector<int> degrees(g.numNodes());
  std::vector<float> nodeGrads(g.numNodes(), 0.0);
  std::vector<float> arcGrads(g.numArcs(), 0.0);
  for (auto n = 0; n < g.numNodes(); ++n) {
    degrees[n] = g.numOut(n);
  }
  float curScore = 0.0;
  float denom = tropical ? 0.0 : std::exp(output - maxScoresCache.back());
  for (auto n : g.accept()) {
    if (g.numOut(n) == 0) {
      computed.push(n);
    }
    if (tropical) {
      curScore = (n == maxArcIdxCache.back()) ? 1.0 : 0.0;
    } else {
      curScore = std::exp(nodeScores[n] - maxScoresCache.back()) / denom;
    }
    nodeGrads[n] += curScore;
  }

  while (!computed.empty()) {
    auto n = computed.front();
    computed.pop();
    denom = tropical ? 0.0 : std::exp(nodeScores[n] - maxScoresCache[n]);
    for (const auto a : g.in(n)) {
      auto un = g.srcNode(a);
      if (tropical) {
        curScore = (a == maxArcIdxCache[n]) ? nodeGrads[n] : 0.0;
      } else {
        curScore = nodeGrads[n] *
            std::exp(nodeScores[un] + g.weight(a) - maxScoresCache[n]) / denom;
      }
      nodeGrads[un] += curScore;
      arcGrads[a] = curScore * deltas.item();
      if ((--degrees[un]) == 0) {
        computed.push(un);
      }
    }
  }
  g.addGrad(std::move(arcGrads));
}

} // namespace

Graph shortestDistance(const Graph& g, bool tropical /* = false */) {
  std::queue<int> computed;
  // List of scores and list of in degrees for each node
  std::vector<float> scores(g.numNodes());
  std::vector<float> maxScoresCache(g.numNodes() + 1, kNegInf);
  std::vector<int> maxArcIdxCache(g.numNodes() + 1, -1);
  std::vector<int> degrees;
  degrees.reserve(g.numNodes());
  for (auto n = 0; n < g.numNodes(); ++n) {
    degrees[n] = g.numIn(n);
  }
  for (auto n : g.start()) {
    if (g.numIn(n) == 0) {
      computed.push(n);
    }
  }

  auto getScore = [tropical](const std::vector<float>& in, float maxScore) {
    if (in.empty()) {
      return kNegInf;
    }
    if (tropical || maxScore == kInf || maxScore == kNegInf) {
      return maxScore;
    }
    float score = -1.0;
    for (auto s : in) {
      score += std::exp(s - maxScore);
    }
    return maxScore + std::log1p(score);
  };

  std::vector<float> inScores;
  float maxScore = kNegInf;
  while (!computed.empty()) {
    auto n = computed.front();
    computed.pop();
    for (auto a : g.in(n)) {
      auto un = g.srcNode(a);
      inScores.push_back(scores[un] + g.weight(a));
      if (inScores.back() > maxScoresCache[n]) {
        maxScoresCache[n] = inScores.back();
        maxArcIdxCache[n] = a;
      }
    }
    if (g.isStart(n)) {
      inScores.push_back(0.0);
      if (inScores.back() > maxScoresCache[n]) {
        maxScoresCache[n] = inScores.back();
        maxArcIdxCache[n] = -1; // an invalid value
      }
    }
    scores[n] = getScore(inScores, maxScoresCache[n]);
    inScores.clear();
    maxScore = kNegInf;
    for (auto a : g.out(n)) {
      auto dn = g.dstNode(a);
      if ((--degrees[dn]) == 0) {
        computed.push(dn);
      }
    }
  }

  // Accumulate scores at all the accept nodes.
  for (auto n : g.accept()) {
    if (degrees[n] > 0) {
      throw std::invalid_argument(
          "Graph has a cycle, self-loop or is disconnected!");
    }
    inScores.push_back(scores[n]);
    if (inScores.back() > maxScoresCache.back()) {
      maxScoresCache.back() = std::max(maxScoresCache.back(), inScores.back());
      maxArcIdxCache.back() = n; // NOTE: Using node idx (instead of arc idx)
    }
  }
  auto score = getScore(inScores, maxScoresCache.back());

  // clear cache not required for bwd
  if (tropical) {
    maxScoresCache.clear();
  } else {
    maxArcIdxCache.clear();
  }

  auto gradFunc = [scores = std::move(scores),
                   maxScoresCache = std::move(maxScoresCache),
                   maxArcIdxCache = std::move(maxArcIdxCache),
                   output = score,
                   tropical](std::vector<Graph>& inputs, Graph deltas) mutable {
    shortestDistanceGrad(
        inputs[0],
        output,
        deltas,
        scores,
        maxScoresCache,
        maxArcIdxCache,
        tropical);
  };

  Graph result(gradFunc, {g});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, score);
  return result;
}

Graph shortestPath(const Graph& g) {
  std::queue<int> computed;
  // List of in degrees for each node
  std::vector<int> degrees;
  degrees.reserve(g.numNodes());
  // List of scores and backpointers for each node
  std::vector<int> backPointers(g.numNodes());
  std::vector<float> scores(g.numNodes(), kNegInf);
  for (auto n = 0; n < g.numNodes(); ++n) {
    degrees[n] = g.numIn(n);
  }
  for (auto n : g.start()) {
    scores[n] = 0.0;
    backPointers[n] = -1;
    if (g.numIn(n) == 0) {
      computed.push(n);
    }
  }

  while (!computed.empty()) {
    auto n = computed.front();
    computed.pop();
    auto score = scores[n];
    for (auto a : g.out(n)) {
      auto dn = g.dstNode(a);
      auto nScore = score + g.weight(a);
      if (nScore > scores[dn]) {
        scores[dn] = nScore;
        backPointers[dn] = a;
      }
      if ((--degrees[dn]) == 0) {
        computed.push(dn);
      }
    }
  }

  // Accumulate scores at all the accept nodes.
  float score = kNegInf;
  int best = -1;
  for (auto a : g.accept()) {
    if (degrees[a] > 0) {
      throw std::invalid_argument(
          "Graph has a cycle, self-loop or is disconnected!");
    }
    if (scores[a] > score) {
      score = scores[a];
      best = a;
    }
  }

  // Chase the pointers to get the best path (backwards)
  std::vector<int> arcs;
  while (best != -1 && backPointers[best] != -1) {
    auto arc = backPointers[best];
    best = g.srcNode(arc);
    arcs.push_back(arc);
  }

  // Build the best path
  Graph out(nullptr, {g});
  if (best != -1) {
    out.addNode(true, arcs.size() == 0);
  }
  for (auto i = arcs.size(); i > 0; --i) {
    out.addNode(false, i == 1);
    out.addArc(
        arcs.size() - i,
        arcs.size() - i + 1,
        g.ilabel(arcs[i - 1]),
        g.olabel(arcs[i - 1]),
        g.weight(arcs[i - 1]));
  }

  auto gradFunc = [arcs = std::move(arcs)](
                      std::vector<Graph>& inputs, Graph deltas) mutable {
    std::vector<float> grad(inputs[0].numArcs(), 0.0);
    for (auto a = 0; a < deltas.numArcs(); ++a) {
      grad[arcs[a]] += deltas.weight(a);
    }
    inputs[0].addGrad(grad);
  };
  out.setGradFunc(std::move(gradFunc));
  return out;
}

} // namespace detail
} // namespace gtn
