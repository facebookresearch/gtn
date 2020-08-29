#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include "gtn/functions/shortest.h"

namespace gtn {
namespace detail {

namespace {

static const float inf = std::numeric_limits<float>::infinity();
static const float neginf = -std::numeric_limits<float>::infinity();

inline float logadd(float a, float b) {
  if (a == neginf) {
    return b;
  }
  if (b == neginf) {
    return a;
  }
  return std::max(a, b) + std::log1p(std::exp(-std::abs(a - b)));
}

void shortestDistanceGrad(
    Graph& g,
    float output,
    const Graph& deltas,
    const std::vector<float>& nodeScores,
    bool tropical) {
  std::queue<int> computed;
  std::vector<int> degrees(g.numNodes());
  std::vector<float> nodeGrads(g.numNodes(), 0.0);
  std::vector<float> arcGrads(g.numArcs(), 0.0);
  for (auto n = 0; n < g.numNodes(); ++n) {
    degrees[n] = g.numOut(n);
  }
  auto updateGrad = [tropical, &nodeGrads, &arcGrads, &deltas](
                        std::vector<std::pair<float, std::pair<int, int>>>& in,
                        float scale) {
    // NB: Perf could be improved by passing max val, idx to the function
    // directly
    if (in.empty()) {
      return;
    }
    auto maxIt = std::max_element(in.begin(), in.end());
    auto maxVal = maxIt->first;
    auto maxIdx = std::distance(in.begin(), maxIt);
    if (tropical) {
      auto& n = in[maxIdx];
      if (n.second.first >= 0) { // node grad
        nodeGrads[n.second.first] += scale;
      }
      if (n.second.second >= 0) { // arc grad
        arcGrads[n.second.second] = scale * deltas.item();
      }
    } else {
      float denom = 0.0;
      for (auto& n : in) {
        n.first = std::exp(n.first - maxVal);
        denom += n.first;
      }
      for (auto& n : in) {
        n.first = scale * (n.first / denom);
        if (n.second.first >= 0) { // node grad
          nodeGrads[n.second.first] += n.first;
        }
        if (n.second.second >= 0) { // arc grad
          arcGrads[n.second.second] = n.first * deltas.item();
        }
      }
    }
  };

  // {score, {node id, arc id}}
  std::vector<std::pair<float, std::pair<int, int>>> inGrads;
  for (auto n : g.accept()) {
    inGrads.emplace_back(nodeScores[n], std::make_pair(n, -1));
    if (g.numOut(n) == 0) {
      computed.push(n);
    }
  }
  updateGrad(inGrads, 1.0);
  inGrads.clear();

  while (!computed.empty()) {
    auto n = computed.front();
    computed.pop();
    for (auto a : g.in(n)) {
      auto un = g.srcNode(a);
      inGrads.emplace_back(nodeScores[un] + g.weight(a), std::make_pair(un, a));
      if ((--degrees[un]) == 0) {
        computed.push(un);
      }
    }
    if (g.start(n)) {
      inGrads.emplace_back(0.0, std::make_pair(-1, -1));
    }
    updateGrad(inGrads, nodeGrads[n]);
    inGrads.clear();
  }
  g.addGrad(std::move(arcGrads));
}

} // namespace

Graph shortestDistance(const Graph& g, bool tropical /* = false */) {
  std::queue<int> computed;
  // List of scores and list of in degrees for each node
  std::vector<float> scores(g.numNodes());
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

  auto getScore = [tropical](const std::vector<float>& in) {
    // NB: Perf could be improved by passing max val to the function directly
    if (in.empty()) {
      return neginf;
    }
    auto maxScore = *std::max_element(in.begin(), in.end());
    if (tropical || maxScore == inf || maxScore == -inf) {
      return maxScore;
    }
    float score = -1.0;
    for (auto s : in) {
      score += std::exp(s - maxScore);
    }
    return maxScore + std::log1p(score);
  };

  std::vector<float> inScores;
  while (!computed.empty()) {
    auto n = computed.front();
    computed.pop();
    for (auto a : g.in(n)) {
      auto un = g.srcNode(a);
      inScores.push_back(scores[un] + g.weight(a));
    }
    if (g.start(n)) {
      inScores.push_back(0.0);
    }
    scores[n] = getScore(inScores);
    inScores.clear();
    for (auto a : g.out(n)) {
      auto dn = g.dstNode(a);
      if ((--degrees[dn]) == 0) {
        computed.push(dn);
      }
    }
  }

  // Accumulate scores at all the accept nodes.
  for (auto a : g.accept()) {
    if (degrees[a] > 0) {
      throw std::invalid_argument(
          "Graph has a cycle, self-loop or is disconnected!");
    }
    inScores.push_back(scores[a]);
  }
  auto score = getScore(inScores);

  auto gradFunc = [scores = std::move(scores), output = score, tropical](
                      std::vector<Graph>& inputs, Graph deltas) mutable {
    shortestDistanceGrad(inputs[0], output, deltas, scores, tropical);
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
  std::vector<float> scores(g.numNodes(), neginf);
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
  float score = neginf;
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
