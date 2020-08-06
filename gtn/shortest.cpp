#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include "shortest.h"

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
    Graph input,
    float output,
    const Graph& deltas,
    std::vector<float>& scores,
    bool tropical) {
  std::queue<int> computed;
  std::vector<int> degrees;
  degrees.reserve(input.numNodes());
  std::vector<float> nodeGrads(input.numNodes(), 0.0);
  std::vector<float> arcGrads(input.numArcs());
  for (auto n = 0; n < input.numNodes(); ++n) {
    degrees[n] = input.numOut(n);
  }
  for (auto n : input.accept()) {
    if (tropical) {
      nodeGrads[n] = deltas.item() * (scores[n] >= output);
    } else {
      nodeGrads[n] = deltas.item() * std::exp(scores[n] - output);
    }
    if (input.numOut(n) == 0) {
      computed.push(n);
    }
  }

  while (!computed.empty()) {
    auto n = computed.front();
    computed.pop();
    auto score = scores[n];
    auto gradn = nodeGrads[n];
    for (auto a : input.in(n)) {
      auto un = input.upNode(a);
      if (tropical) {
        if ((input.weight(a) + scores[un]) >= score) {
          arcGrads[a] = gradn;
          nodeGrads[un] = gradn;
        } else {
          arcGrads[a] = 0.0;
        }
      } else {
        arcGrads[a] = gradn * std::exp(input.weight(a) + scores[un] - score);
        nodeGrads[un] += arcGrads[a];
      }
      if ((--degrees[un]) == 0) {
        computed.push(un);
      }
    }
  }
  input.addGrad(std::move(arcGrads));
}

} // namespace

Graph shortestDistance(const Graph& graph, bool tropical /* = false */) {
  std::queue<int> computed;
  // List of scores and list of in degrees for each node
  std::vector<float> scores(graph.numNodes());
  std::vector<int> degrees;
  degrees.reserve(graph.numNodes());
  for (auto n = 0; n < graph.numNodes(); ++n) {
    degrees[n] = graph.numIn(n);
  }
  for (auto n : graph.start()) {
    if (graph.numIn(n) == 0) {
      computed.push(n);
    }
  }

  auto getScore = [tropical](const std::vector<float>& in) {
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
    for (auto a : graph.in(n)) {
      auto un = graph.upNode(a);
      inScores.push_back(scores[un] + graph.weight(a));
    }
    if (graph.start(n)) {
      inScores.push_back(0.0);
    }
    scores[n] = getScore(inScores);
    inScores.clear();
    for (auto a : graph.out(n)) {
      auto dn = graph.downNode(a);
      if ((--degrees[dn]) == 0) {
        computed.push(dn);
      }
    }
  }

  // Accumulate scores at all the accept nodes.
  for (auto a : graph.accept()) {
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

  Graph result(gradFunc, {graph});
  result.addNode(true);
  result.addNode(false, true);
  result.addArc(0, 1, 0, 0, score);
  return result;
}

Graph shortestPath(const Graph& graph) {
  std::queue<int> computed;
  // List of in degrees for each node
  std::vector<int> degrees;
  degrees.reserve(graph.numNodes());
  // List of scores and backpointers for each node
  std::vector<int> backPointers(graph.numNodes());
  std::vector<float> scores(graph.numNodes(), neginf);
  for (auto n = 0; n < graph.numNodes(); ++n) {
    degrees[n] = graph.numIn(n);
  }
  for (auto n : graph.start()) {
    scores[n] = 0.0;
    backPointers[n] = -1;
    if (graph.numIn(n) == 0) {
      computed.push(n);
    }
  }

  while (!computed.empty()) {
    auto n = computed.front();
    computed.pop();
    auto score = scores[n];
    for (auto a : graph.out(n)) {
      auto dn = graph.downNode(a);
      auto nScore = score + graph.weight(a);
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
  for (auto a : graph.accept()) {
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
    best = graph.upNode(arc);
    arcs.push_back(arc);
  }

  // Build the best path
  Graph out;
  if (best != -1) {
    out.addNode(true, arcs.size() == 0);
  }
  for (auto i = arcs.size(); i > 0; --i) {
    out.addNode(false, i == 1);
    out.addArc(
        arcs.size() - i,
        arcs.size() - i + 1,
        graph.ilabel(arcs[i - 1]),
        graph.olabel(arcs[i - 1]),
        graph.weight(arcs[i - 1]));
  }

  auto gradFunc = [arcs = std::move(arcs)](
                      std::vector<Graph>& inputs, Graph deltas) mutable {
    std::vector<float> grad(inputs[0].numArcs(), 0.0);
    for (auto a = 0; a < deltas.numArcs(); ++a) {
      grad[arcs[a]] += deltas.weight(a);
    }
    inputs[0].addGrad(grad);
  };
  return Graph(out, gradFunc, {graph.withoutWeights()});
}

} // namespace detail
} // namespace gtn
