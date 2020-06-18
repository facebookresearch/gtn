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

using NodeMap = std::unordered_map<std::pair<int, int>, Node*, hashIntPair>;
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

bool equals(Graph a, Graph b) {
  if (a.numNodes() != b.numNodes() || a.numStart() != b.numStart() ||
      a.numAccept() != b.numAccept() || a.numArcs() != b.numArcs()) {
    return false;
  }

  for (size_t n = 0; n < a.numNodes(); n++) {
    auto aNode = a.node(n);
    auto bNode = b.node(n);
    if (aNode->numIn() != bNode->numIn() ||
        aNode->numOut() != bNode->numOut() ||
        aNode->start() != bNode->start() ||
        aNode->accept() != bNode->accept()) {
      return false;
    }

    std::list<Arc*> bOut(bNode->out().begin(), bNode->out().end());
    for (auto arcA : aNode->out()) {
      auto it = bOut.begin();
      for (; it != bOut.end(); it++) {
        auto arcB = *it;
        if (arcA->downNode()->index() == arcB->downNode()->index() &&
            arcA->upNode()->index() == arcB->upNode()->index() &&
            arcA->ilabel() == arcB->ilabel() &&
            arcA->olabel() == arcB->olabel() &&
            arcA->weight() == arcB->weight()) {
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

bool isomorphic(Node* a, Node* b, NodeMap& visited) {
  auto state = std::make_pair(a->index(), b->index());
  // We assume a state is good unless found to be otherwise
  auto item = visited.insert({state, a});
  if (!item.second) {
    return item.first->second != nullptr;
  }

  if (a->numIn() != b->numIn() || a->numOut() != b->numOut() ||
      a->start() != b->start() || a->accept() != b->accept()) {
    item.first->second = nullptr;
    return false;
  }

  // Each arc in a has to match with an arc in b
  std::list<Arc*> bOut(b->out().begin(), b->out().end());
  for (auto outa : a->out()) {
    auto it = bOut.begin();
    for (; it != bOut.end(); it++) {
      auto outb = *it;
      if (outa->ilabel() != outb->ilabel() ||
          outa->olabel() != outb->olabel() ||
          outa->weight() != outb->weight()) {
        continue;
      }
      if (isomorphic(outa->downNode(), outb->downNode(), visited)) {
        break;
      }
    }
    if (it == bOut.end()) {
      item.first->second = nullptr;
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
      isIsomorphic |= isomorphic(s1, s2, visited);
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
  auto startNodes = graph.start();
  if (startNodes.size() > 0) {
    out << startNodes[0]->index();
    for (size_t i = 1; i < startNodes.size(); i++) {
      out << " " << startNodes[i]->index();
    }
    out << std::endl;
  }

  // Print accept nodes
  auto acceptNodes = graph.accept();
  if (acceptNodes.size() > 0) {
    out << acceptNodes[0]->index();
    for (size_t i = 1; i < acceptNodes.size(); i++) {
      out << " " << acceptNodes[i]->index();
    }
    out << std::endl;
  }

  // Print arcs
  for (auto& a : graph.arcs()) {
    out << a.upNode()->index() << " " << a.downNode()->index() << " "
        << a.ilabel() << " " << a.olabel() << " " << a.weight() << std::endl;
  }
}

void draw(
    Graph graph,
    std::ostream& out,
    const SymbolMap& isymbols /* = SymbolMap() */,
    const SymbolMap& osymbols /* = SymbolMap() */) {
  out << "digraph FST {\n  rankdir = LR;\n  label = \"\";\n"
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

  auto drawNode = [acceptor = graph.acceptor(),
                   &out,
                   &isymbols,
                   &osymbols,
                   &label_to_string](auto node) {
    std::string style = node->start() || node->accept() ? "bold" : "solid";
    std::string shape = node->accept() ? "doublecircle" : "circle";
    out << "  " << node->index() << " [label = \"" << node->index()
        << "\", shape = " << shape << ", style = " << style
        << ", fontsize = 14];\n";
    for (auto arc : node->out()) {
      auto ilabel = label_to_string(isymbols, arc->ilabel());
      out << "  " << arc->upNode()->index() << " -> "
          << arc->downNode()->index() << " [label = \"" << ilabel;
      if (!acceptor || !osymbols.empty()) {
        auto olabel = label_to_string(osymbols, arc->olabel());
        out << ":" << olabel;
      }
      out << "/" << arc->weight() << "\", fontsize = 14];\n";
    }
  };

  // Draw start nodes first and accept nodes last to help with layout.
  for (auto node : graph.start()) {
    drawNode(node);
  }

  for (auto& node : graph.nodes()) {
    if (!node.start() && !node.accept()) {
      drawNode(&node);
    }
  }

  for (auto node : graph.accept()) {
    if (node->start()) {
      continue;
    }
    drawNode(node);
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
