#include "utils.h"

#include <list>
#include <unordered_set>

namespace gtn {

namespace {
struct hashIntPair {
  size_t operator()(const std::pair<int, int>& p) const {
    return (p.first * (static_cast<size_t>(INT_MAX) + 1) + p.second);
  }
};

using NodeMap = std::unordered_map<std::pair<int, int>, int, hashIntPair>;
static const std::string epsilonSymbol = "ε";
} // namespace

static std::vector<std::string> split(const std::string& input) {
  std::vector<std::string> result;
  size_t i = 0;
  while (true) {
    auto j = input.find(" ", i);
    if (j == std::string::npos) {
      break;
    }
    result.emplace_back(input.begin() + i, input.begin() + j);
    i = j + 1;
  }
  result.emplace_back(input.begin() + i, input.end());
  return result;
}

bool equal(Graph a, Graph b) {
  if (a.numNodes() != b.numNodes() || a.numStart() != b.numStart() ||
      a.numAccept() != b.numAccept() || a.numArcs() != b.numArcs()) {
    return false;
  }

  for (size_t n = 0; n < a.numNodes(); n++) {
    if (a.numIn(n) != b.numIn(n) || a.numOut(n) != b.numOut(n) ||
        a.start(n) != b.start(n) || a.accept(n) != b.accept(n)) {
      return false;
    }

    std::list<int> bOut(b.out(n).begin(), b.out(n).end());
    for (auto arcA : a.out(n)) {
      auto it = bOut.begin();
      for (; it != bOut.end(); it++) {
        auto arcB = *it;
        if (a.downNode(arcA) == b.downNode(arcB) &&
            a.upNode(arcA) == b.upNode(arcB) &&
            a.ilabel(arcA) == b.ilabel(arcB) &&
            a.olabel(arcA) == b.olabel(arcB) &&
            a.weight(arcA) == b.weight(arcB)) {
          break;
        }
      }
      if (it == bOut.end()) {
        return false;
      }
      bOut.erase(it);
    }
  }
  return true;
}

bool isomorphic(Graph& a, Graph& b, int aNode, int bNode, NodeMap& visited) {
  auto state = std::make_pair(aNode, bNode);
  // We assume a state is good unless found to be otherwise
  auto item = visited.insert({state, aNode});
  if (!item.second) {
    return item.first->second >= 0;
  }

  if (a.numIn(aNode) != b.numIn(bNode) || a.numOut(aNode) != b.numOut(bNode) ||
      a.start(aNode) != b.start(bNode) || a.accept(aNode) != b.accept(bNode)) {
    item.first->second = -1;
    return false;
  }

  // Each arc in a has to match with an arc in b
  std::list<int> bOut(b.out(bNode).begin(), b.out(bNode).end());
  for (auto aArc : a.out(aNode)) {
    auto it = bOut.begin();
    for (; it != bOut.end(); it++) {
      auto bArc = *it;
      if (a.ilabel(aArc) != b.ilabel(bArc) ||
          a.olabel(aArc) != b.olabel(bArc) ||
          a.weight(aArc) != b.weight(bArc)) {
        continue;
      }
      if (isomorphic(a, b, a.downNode(aArc), b.downNode(bArc), visited)) {
        break;
      }
    }
    if (it == bOut.end()) {
      item.first->second = -1;
      return false;
    }
    bOut.erase(it);
  }

  // If we get here then we return true
  return true;
}

bool isomorphic(Graph a, Graph b) {
  if (a.numNodes() != b.numNodes() || a.numStart() != b.numStart() ||
      a.numAccept() != b.numAccept() || a.numArcs() != b.numArcs()) {
    return false;
  }

  bool isIsomorphic = false;
  NodeMap visited;
  for (auto s1 : a.start()) {
    for (auto s2 : b.start()) {
      isIsomorphic |= isomorphic(a, b, s1, s2, visited);
    }
  }
  return isIsomorphic;
}

Graph load(const std::string& fileName) {
  std::ifstream in;
  in.open(fileName);
  if (!in) {
    throw std::invalid_argument("Couldn't find graph file to load.");
  }
  return load(in);
}

Graph load(std::istream&& in) {
  return load(in);
}

Graph load(std::istream& in) {
  std::unordered_map<int, int> label_to_idx;

  // For now we don't allow loading an empty graph
  // or a graph without any start and accept nodes.
  auto getNodes = [&in](std::string sora) {
    std::string line;
    if (!std::getline(in, line)) {
      throw std::invalid_argument("Must specify " + sora + " node(s).");
    }
    auto nodes = split(line);
    if (nodes.size() == 0) {
      throw std::invalid_argument("Must specify " + sora + " node(s).");
    }

    std::unordered_set<int> nodeSet;
    for (auto& n : nodes) {
      nodeSet.insert(std::stoi(n));
    }
    if (nodeSet.size() != nodes.size()) {
      throw std::invalid_argument("Repeat " + sora + " node detected.");
    }
    return nodeSet;
  };

  Graph graph;
  auto startNodes = getNodes("start");
  auto acceptNodes = getNodes("accept");
  for (auto s : startNodes) {
    graph.addNode(true, acceptNodes.count(s));
    label_to_idx[s] = graph.numNodes() - 1;
  }
  for (auto a : acceptNodes) {
    // An accept node can also be a start node
    if (!startNodes.count(a)) {
      graph.addNode(false, true);
      label_to_idx[a] = graph.numNodes() - 1;
    }
  }

  std::string line;
  while (std::getline(in, line)) {
    // Add the arc in the line
    auto tokens = split(line);
    if (tokens.size() < 3 || tokens.size() > 5) {
      throw std::invalid_argument("Bad line for loading arc.");
    }
    auto src = std::stoi(tokens[0]);
    auto dst = std::stoi(tokens[1]);
    if (!label_to_idx.count(src)) {
      graph.addNode();
      label_to_idx[src] = graph.numNodes() - 1;
    }
    if (!label_to_idx.count(dst)) {
      graph.addNode();
      label_to_idx[dst] = graph.numNodes() - 1;
    }
    if (tokens.size() == 5) {
      graph.addArc(
          label_to_idx[src],
          label_to_idx[dst],
          std::stoi(tokens[2]),
          std::stoi(tokens[3]),
          std::stof(tokens[4]));
    } else if (tokens.size() == 4) {
      graph.addArc(
          label_to_idx[src],
          label_to_idx[dst],
          std::stoi(tokens[2]),
          std::stoi(tokens[3]));
    } else {
      graph.addArc(label_to_idx[src], label_to_idx[dst], std::stoi(tokens[2]));
    }
  }
  return graph;
}

void print(Graph graph) {
  print(graph, std::cout);
}

void print(Graph graph, std::ostream& out) {
  // Print start nodes
  auto& startNodes = graph.start();
  if (startNodes.size() > 0) {
    out << startNodes[0];
    for (size_t i = 1; i < startNodes.size(); i++) {
      out << " " << startNodes[i];
    }
    out << std::endl;
  }

  // Print accept nodes
  auto& acceptNodes = graph.accept();
  if (acceptNodes.size() > 0) {
    out << acceptNodes[0];
    for (size_t i = 1; i < acceptNodes.size(); i++) {
      out << " " << acceptNodes[i];
    }
    out << std::endl;
  }

  // Print arcs
  for (int i = 0; i < graph.numArcs(); i++) {
    out << graph.upNode(i) << " " << graph.downNode(i) << " " << graph.ilabel(i)
        << " " << graph.olabel(i) << " " << graph.weight(i) << std::endl;
  }
}

void draw(
    Graph graph,
    std::ostream& out,
    const SymbolMap& isymbols /* = SymbolMap() */,
    const SymbolMap& osymbols /* = SymbolMap() */) {
  out << "digraph FST {\n  margin = 0;\n  rankdir = LR;\n  label = \"\";\n"
      << "  center = 1;\n  ranksep = \"0.4\";\n  nodesep = \"0.25\";\n";

  auto label_to_string = [](const auto& symbols, const auto label) {
    if (symbols.empty()) {
      return std::to_string(label);
    }
    if (label == Graph::epsilon) {
      return epsilonSymbol;
    }
    return symbols.at(label);
  };

  auto drawNode = [&graph, &out, &isymbols, &osymbols, &label_to_string](
                      auto n) {
    std::string penwidth = graph.start(n) ? "2.0" : "1.0";
    std::string shape = graph.accept(n) ? "doublecircle" : "circle";
    out << "  " << n << " [label = \"" << n << "\", shape = " << shape
        << ", penwidth = " << penwidth << ", fontsize = 14];\n";
    for (auto a : graph.out(n)) {
      auto ilabel = label_to_string(isymbols, graph.ilabel(a));
      out << "  " << graph.upNode(a) << " -> " << graph.downNode(a)
          << " [label = \"" << ilabel;
      if (!osymbols.empty()) {
        auto olabel = label_to_string(osymbols, graph.olabel(a));
        out << ":" << olabel;
      }
      out << "/" << graph.weight(a) << "\", fontsize = 14];\n";
    }
  };

  // Draw start nodes first and accept nodes last to help with layout.
  for (auto i : graph.start()) {
    drawNode(i);
  }

  for (int i = 0; i < graph.numNodes(); i++) {
    if (!graph.start(i) && !graph.accept(i)) {
      drawNode(i);
    }
  }

  for (auto i : graph.accept()) {
    if (graph.start(i)) {
      continue;
    }
    drawNode(i);
  }

  out << "}";
}

void draw(
    Graph graph,
    const std::string& filename,
    const SymbolMap& isymbols /* = SymbolMap() */,
    const SymbolMap& osymbols /* = SymbolMap() */) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    throw std::runtime_error("Could not open file [" + filename + "]");
  }
  draw(graph, out, isymbols, osymbols);
}

} // namespace gtn
