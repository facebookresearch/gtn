#include "gtn/utils.h"

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

bool equal(const Graph& g1, const Graph& g2) {
  if (g1.numNodes() != g2.numNodes() || g1.numStart() != g2.numStart() ||
      g1.numAccept() != g2.numAccept() || g1.numArcs() != g2.numArcs()) {
    return false;
  }

  for (size_t n = 0; n < g1.numNodes(); n++) {
    if (g1.numIn(n) != g2.numIn(n) || g1.numOut(n) != g2.numOut(n) ||
        g1.start(n) != g2.start(n) || g1.accept(n) != g2.accept(n)) {
      return false;
    }

    std::list<int> bOut(g2.out(n).begin(), g2.out(n).end());
    for (auto arcG1 : g1.out(n)) {
      auto it = bOut.begin();
      for (; it != bOut.end(); it++) {
        auto arcG2 = *it;
        if (g1.dstNode(arcG1) == g2.dstNode(arcG2) &&
            g1.srcNode(arcG1) == g2.srcNode(arcG2) &&
            g1.ilabel(arcG1) == g2.ilabel(arcG2) &&
            g1.olabel(arcG1) == g2.olabel(arcG2) &&
            g1.weight(arcG1) == g2.weight(arcG2)) {
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

bool isomorphic(
    const Graph& g1,
    const Graph& g2,
    int g1Node,
    int g2Node,
    NodeMap& visited) {
  auto state = std::make_pair(g1Node, g2Node);
  // We assume a state is good unless found to be otherwise
  auto item = visited.insert({state, g1Node});
  if (!item.second) {
    return item.first->second >= 0;
  }

  if (g1.numIn(g1Node) != g2.numIn(g2Node) ||
      g1.numOut(g1Node) != g2.numOut(g2Node) ||
      g1.start(g1Node) != g2.start(g2Node) ||
      g1.accept(g1Node) != g2.accept(g2Node)) {
    item.first->second = -1;
    return false;
  }

  // Each arc in a has to match with an arc in b
  std::list<int> bOut(g2.out(g2Node).begin(), g2.out(g2Node).end());
  for (auto aArc : g1.out(g1Node)) {
    auto it = bOut.begin();
    for (; it != bOut.end(); it++) {
      auto bArc = *it;
      if (g1.ilabel(aArc) != g2.ilabel(bArc) ||
          g1.olabel(aArc) != g2.olabel(bArc) ||
          g1.weight(aArc) != g2.weight(bArc)) {
        continue;
      }
      if (isomorphic(g1, g2, g1.dstNode(aArc), g2.dstNode(bArc), visited)) {
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

bool isomorphic(const Graph& g1, const Graph& g2) {
  if (g1.numNodes() != g2.numNodes() || g1.numStart() != g2.numStart() ||
      g1.numAccept() != g2.numAccept() || g1.numArcs() != g2.numArcs()) {
    return false;
  }

  bool isIsomorphic = false;
  NodeMap visited;
  for (auto s1 : g1.start()) {
    for (auto s2 : g2.start()) {
      isIsomorphic |= isomorphic(g1, g2, s1, s2, visited);
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

void print(const Graph& g) {
  print(g, std::cout);
}

void print(const Graph& g, std::ostream& out) {
  // Print start nodes
  auto& startNodes = g.start();
  if (startNodes.size() > 0) {
    out << startNodes[0];
    for (size_t i = 1; i < startNodes.size(); i++) {
      out << " " << startNodes[i];
    }
    out << std::endl;
  }

  // Print accept nodes
  auto& acceptNodes = g.accept();
  if (acceptNodes.size() > 0) {
    out << acceptNodes[0];
    for (size_t i = 1; i < acceptNodes.size(); i++) {
      out << " " << acceptNodes[i];
    }
    out << std::endl;
  }

  // Print arcs
  for (int i = 0; i < g.numArcs(); i++) {
    out << g.srcNode(i) << " " << g.dstNode(i) << " " << g.ilabel(i) << " "
        << g.olabel(i) << " " << g.weight(i) << std::endl;
  }
}

void draw(
    const Graph& g,
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

  auto drawNode = [&g, &out, &isymbols, &osymbols, &label_to_string](auto n) {
    std::string penwidth = g.start(n) ? "2.0" : "1.0";
    std::string shape = g.accept(n) ? "doublecircle" : "circle";
    out << "  " << n << " [label = \"" << n << "\", shape = " << shape
        << ", penwidth = " << penwidth << ", fontsize = 14];\n";
    for (auto a : g.out(n)) {
      auto ilabel = label_to_string(isymbols, g.ilabel(a));
      out << "  " << g.srcNode(a) << " -> " << g.dstNode(a) << " [label = \""
          << ilabel;
      if (!osymbols.empty()) {
        auto olabel = label_to_string(osymbols, g.olabel(a));
        out << ":" << olabel;
      }
      out << "/" << g.weight(a) << "\", fontsize = 14];\n";
    }
  };

  // Draw start nodes first and accept nodes last to help with layout.
  for (auto i : g.start()) {
    drawNode(i);
  }

  for (int i = 0; i < g.numNodes(); i++) {
    if (!g.start(i) && !g.accept(i)) {
      drawNode(i);
    }
  }

  for (auto i : g.accept()) {
    if (g.start(i)) {
      continue;
    }
    drawNode(i);
  }

  out << "}";
}

void draw(
    const Graph& g,
    const std::string& filename,
    const SymbolMap& isymbols /* = SymbolMap() */,
    const SymbolMap& osymbols /* = SymbolMap() */) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    throw std::runtime_error("Could not open file [" + filename + "]");
  }
  draw(g, out, isymbols, osymbols);
}

} // namespace gtn
