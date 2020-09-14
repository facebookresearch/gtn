#include "gtn/utils.h"

#include <algorithm>
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
constexpr const int kNumMaxSummary = 10;
constexpr const int kSummaryThreshold = 20;

std::vector<std::string> split(const std::string& input) {
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
} // namespace

bool equal(const Graph& g1, const Graph& g2) {
  if (g1.numNodes() != g2.numNodes() || g1.numStart() != g2.numStart() ||
      g1.numAccept() != g2.numAccept() || g1.numArcs() != g2.numArcs()) {
    return false;
  }

  for (size_t n = 0; n < g1.numNodes(); n++) {
    if (g1.numIn(n) != g2.numIn(n) || g1.numOut(n) != g2.numOut(n) ||
        g1.isStart(n) != g2.isStart(n) || g1.isAccept(n) != g2.isAccept(n)) {
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
      g1.isStart(g1Node) != g2.isStart(g2Node) ||
      g1.isAccept(g1Node) != g2.isAccept(g2Node)) {
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

void save(std::ostream& out, const Graph& g) {
  // save num_nodes, num_arcs,  num_start_nodes, num_accept_nodes
  std::vector<int> nums = {
      g.numNodes(), g.numArcs(), g.numStart(), g.numAccept()};
  out.write(
      reinterpret_cast<const char*>(nums.data()), nums.size() * sizeof(int));

  // save start nodes
  const auto& start = g.start();
  out.write(
      reinterpret_cast<const char*>(start.data()), start.size() * sizeof(int));

  // save accept nodes
  const auto& accept = g.accept();
  out.write(
      reinterpret_cast<const char*>(accept.data()),
      accept.size() * sizeof(int));

  // save arcs
  for (int i = 0; i < g.numArcs(); i++) {
    std::vector<int> indexes = {
        g.srcNode(i), g.dstNode(i), g.ilabel(i), g.olabel(i)};
    out.write(
        reinterpret_cast<const char*>(indexes.data()),
        indexes.size() * sizeof(int));
  }

  // save weights
  const float* weights = g.weights();
  out.write(
      reinterpret_cast<const char*>(weights), g.numArcs() * sizeof(float));
}

Graph load(std::istream& in) {
  // load num_nodes, num_arcs, num_start_nodes, num_accept_nodes
  int numNodes, numArcs, numStart, numAccept;
  in.read(reinterpret_cast<char*>(&numNodes), sizeof(int));
  in.read(reinterpret_cast<char*>(&numArcs), sizeof(int));
  in.read(reinterpret_cast<char*>(&numStart), sizeof(int));
  in.read(reinterpret_cast<char*>(&numAccept), sizeof(int));

  // load start nodes
  std::vector<int> start(numStart);
  in.read(reinterpret_cast<char*>(start.data()), numStart * sizeof(int));

  // load accept nodes
  std::vector<int> accept(numAccept);
  in.read(reinterpret_cast<char*>(accept.data()), numAccept * sizeof(int));

  // create nodes
  Graph g;
  std::unordered_set<int> startSet{start.begin(), start.end()};
  std::unordered_set<int> acceptSet{accept.begin(), accept.end()};
  for (int i = 0; i < numNodes; ++i) {
    g.addNode(startSet.count(i), acceptSet.count(i));
  }

  // create arcs
  std::vector<int> cols(4);
  for (int i = 0; i < numArcs; ++i) {
    in.read(reinterpret_cast<char*>(cols.data()), cols.size() * sizeof(int));
    g.addArc(cols[0], cols[1], cols[2], cols[3]);
  }

  // load weights
  std::vector<float> weights(numArcs);
  in.read(
      reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
  g.setWeights(weights.data());
  return g;
}

void saveTxtImpl(std::ostream& out, const Graph& g, bool limitOutput) {
  // `limitOutput` param can be used to output only a brief summary rather than
  // full representation of the graph.

  // save start nodes
  for (int i = 0; i < g.numStart(); ++i) {
    if (limitOutput && i >= kNumMaxSummary) {
      out << " ...";
      break;
    }
    if (i > 0) {
      out << " ";
    }
    out << g.start()[i];
  }
  out << "\n";

  // save accept nodes
  for (int i = 0; i < g.numAccept(); ++i) {
    if (limitOutput && i >= kNumMaxSummary) {
      out << " ...";
      break;
    }
    if (i > 0) {
      out << " ";
    }
    out << g.accept()[i];
  }
  out << "\n";

  // save arcs
  for (int i = 0; i < g.numArcs(); i++) {
    if (limitOutput && i >= kNumMaxSummary) {
      out << "...\n";
      break;
    }
    out << g.srcNode(i) << " " << g.dstNode(i) << " " << g.ilabel(i) << " "
        << g.olabel(i) << " " << g.weight(i) << "\n";
  }
}

void saveTxt(std::ostream& out, const Graph& g) {
  saveTxtImpl(out, g, false /* limitOutput */);
}

Graph loadTxt(std::istream& in) {
  // load start nodes
  std::string line;
  std::vector<std::string> cols;
  if (!std::getline(in, line)) {
    throw std::invalid_argument("Must specify start node(s).");
  }

  cols = split(line);
  std::vector<int> start(cols.size());
  std::transform(cols.begin(), cols.end(), start.begin(), [](std::string s) {
    return std::stoi(s);
  });
  // load accept nodes
  if (!std::getline(in, line)) {
    throw std::invalid_argument("Must specify accept node(s).");
  }

  cols = split(line);
  std::vector<int> accept(cols.size());
  std::transform(cols.begin(), cols.end(), accept.begin(), [](std::string s) {
    return std::stoi(s);
  });

  // create nodes
  Graph g;
  int maxNodeIdx = -1;
  for (auto s : start) {
    maxNodeIdx = std::max(s, maxNodeIdx);
  }
  for (auto a : accept) {
    maxNodeIdx = std::max(a, maxNodeIdx);
  }

  std::unordered_set<int> startSet{start.begin(), start.end()};
  std::unordered_set<int> acceptSet{accept.begin(), accept.end()};
  if (startSet.size() != start.size()) {
    throw std::invalid_argument("Repeat start node detected.");
  }
  if (acceptSet.size() != accept.size()) {
    throw std::invalid_argument("Repeat accept node detected.");
  }
  for (int i = 0; i <= maxNodeIdx; ++i) {
    g.addNode(startSet.count(i), acceptSet.count(i));
  }

  // create arcs
  while (std::getline(in, line)) {
    // Add the arc in the line
    cols = split(line);
    if (cols.size() < 3 || cols.size() > 5) {
      throw std::invalid_argument("Bad line for loading arc.");
    }
    auto src = std::stoi(cols[0]);
    auto dst = std::stoi(cols[1]);

    if (src > maxNodeIdx || dst > maxNodeIdx) {
      auto newMax = std::max(src, dst);
      for (int i = 0; i < (newMax - maxNodeIdx); ++i) {
        g.addNode();
      }
      maxNodeIdx = newMax;
    }
    if (cols.size() == 5) {
      g.addArc(
          src, dst, std::stoi(cols[2]), std::stoi(cols[3]), std::stof(cols[4]));
    } else if (cols.size() == 4) {
      g.addArc(src, dst, std::stoi(cols[2]), std::stoi(cols[3]));
    } else {
      g.addArc(src, dst, std::stoi(cols[2]));
    }
  }
  return g;
}

void save(const std::string& fileName, const Graph& g) {
  std::ofstream out(fileName, std::ios::binary);
  save(out, g);
}

Graph load(const std::string& fileName) {
  std::ifstream in(fileName, std::ios::binary);
  if (!in) {
    throw std::invalid_argument(
        "Couldn't find graph file to load. '" + fileName + "'");
  }
  return load(in);
}

Graph load(std::istream&& in) {
  return load(in);
}

void saveTxt(
    const std::string& fileName,
    const Graph& g) {
  std::ofstream out(fileName);
  saveTxt(out, g);
}

Graph loadTxt(const std::string& fileName) {
  std::ifstream in(fileName);
  if (!in) {
    throw std::invalid_argument(
        "Couldn't find graph file to load. '" + fileName + "'");
  }
  return loadTxt(in);
}

Graph loadTxt(std::istream&& in) {
  return loadTxt(in);
}

std::ostream& operator<<(std::ostream& out, const Graph& g) {
  saveTxtImpl(out, g, std::max(g.numArcs(), g.numNodes()) > kSummaryThreshold);
  return out;
}

void draw(
    const Graph& g,
    std::ostream& out,
    const SymbolMap& isymbols /* = SymbolMap() */,
    const SymbolMap& osymbols /* = SymbolMap() */) {
  out << "digraph FST {\n  margin = 0;\n  rankdir = LR;\n  label = \"\";\n"
      << "  center = 1;\n  ranksep = \"0.4\";\n  nodesep = \"0.25\";\n";

  auto label_to_string = [](const auto& symbols, const auto label) {
    if (label == Graph::epsilon) {
      return epsilonSymbol;
    }
    if (symbols.empty()) {
      return std::to_string(label);
    }
    return symbols.at(label);
  };

  auto drawNode = [&g, &out, &isymbols, &osymbols, &label_to_string](auto n) {
    std::string penwidth = g.isStart(n) ? "2.0" : "1.0";
    std::string shape = g.isAccept(n) ? "doublecircle" : "circle";
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
    if (!g.isStart(i) && !g.isAccept(i)) {
      drawNode(i);
    }
  }

  for (auto i : g.accept()) {
    if (g.isStart(i)) {
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
