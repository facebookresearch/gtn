#pragma once

#include <cassert>
#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace gtn {

class Graph {
/* Graph operations are in the log or tropical semirings. The default score
 * for an arc is 0 (e.g. the multiplicative identity) and the additive
 * identity is -infinity. Path scores are accumulated with the logadd or
 * max operations and the score for a path is accumulated with addition. */

 private:
  struct Node {
    Node(bool start, bool accept) : start(start), accept(accept){};
    bool start{false};
    bool accept{false};
    std::vector<int> in;
    std::vector<int> out;
  };

  struct Arc {
    Arc(int upNode, int downNode, int ilabel, int olabel)
        : upNode(upNode),
          downNode(downNode),
          ilabel(ilabel),
          olabel(olabel) {};
    int upNode;
    int downNode;
    int ilabel;
    int olabel;
  };

 public:
  using GradFunc =
      std::function<void(std::vector<Graph>& inputs, Graph& deltas)>;

  Graph(GradFunc gradFunc, std::vector<Graph> inputs);
  Graph(Graph& data, GradFunc gradFunc, std::vector<Graph> inputs);
  Graph(bool calcGrad = true);

  int addNode(bool start = false, bool accept = false);

  /* Add an arc between two nodes. */
  int addArc(int upNode, int downNode, int label);
  int addArc(
      int upNode,
      int downNode,
      int ilabel,
      int olabel,
      float weight = 0.0);

  // Attempt to keep code like `g.addArc(n1, n2, 0, 2.0)` from compiling
  int addArc(int upNode, int downNode, int label, float) = delete;
  int addArc(int upNode, int downNode, int label, double) = delete;

  int numArcs() const {
    return sharedGraph_->arcs.size();
  };
  int numNodes() const {
    return sharedGraph_->nodes.size();
  };
  int numStart() const {
    return sharedGraph_->start.size();
  };
  int numAccept() const {
    return sharedGraph_->accept.size();
  };
  bool acceptor() const {
    return sharedGraph_->acceptor;
  }

  /* Get the score on a single arc graph. */
  float item() const;

  void addGrad(std::vector<float>&& other);
  void addGrad(const std::vector<float>& other);
  void addGrad(const Graph& other);

  bool calcGrad() const {
    return sharedGrad_->calcGrad;
  };
  bool isGradAvailable() const {
    return sharedGrad_->grad != nullptr;
  }

  Graph& grad();

  void setCalcGrad(bool calcGrad);
  void zeroGrad();
  std::uintptr_t id();
  GradFunc gradFunc() {
    return sharedGrad_->gradFunc;
  };
  std::vector<Graph>& inputs() {
    return sharedGrad_->inputs;
  };

  /* A deep copy of a graph `other` which is not recorded in the
   * autograd tape. For a version which is recorded in the
   * autograd tape see `clone`. */
  static Graph deepCopy(const Graph& src);

  static constexpr int epsilon{-1};

  // Accessing and modifying nodes.

  const std::vector<int>& start() const {
    return sharedGraph_->start;
  };
  const std::vector<int>& accept() const {
    return sharedGraph_->accept;
  };
  bool start(int i) const {
    return node(i)->start;
  };
  bool accept(int i) const {
    return node(i)->accept;
  };
  void makeAccept(int i) {
    auto n = node(i);
    if (!n->accept) {
      sharedGraph_->accept.push_back(i);
      n->accept = true;
    }
  };
  int numOut(int i) const {
    return node(i)->out.size();
  }
  const std::vector<int>& out(int i) const {
    return node(i)->out;
  }
  int out(int i, int j) const {
    return node(i)->out[j];
  }
  int numIn(int i) const {
    return node(i)->in.size();
  }
  const std::vector<int>& in(int i) const {
    return node(i)->in;
  }
  int in(int i, int j) const {
    return node(i)->in[j];
  }

  // Accessing and modifying arcs.

  int upNode(int i) const {
    return arc(i)->upNode;
  }
  int downNode(int i) const {
    return arc(i)->downNode;
  }
  int label(int i) const {
    return arc(i)->ilabel;
  }
  int ilabel(int i) const {
    return arc(i)->ilabel;
  }
  int olabel(int i) const {
    return arc(i)->olabel;
  }
  float weight(int i) const {
    return (*sharedWeights_)[i];
  }
  void setWeight(int i, float weight) {
    (*sharedWeights_)[i] = weight;
  }

 private:
  const Node* node(int i) const {
    // NB: assert gets stripped at in release mode
    assert(i >= 0 && i < numNodes());
    return &sharedGraph_->nodes[i];
  }
  Node* node(int i) {
    return const_cast<Node*>(static_cast<const Graph&>(*this).node(i));
  }
  const Arc* arc(int i) const {
    // NB: assert gets stripped at in release mode
    assert(i >= 0 && i < numArcs());
    return &sharedGraph_->arcs[i];
  }
  Arc* arc(int i) {
    return const_cast<Arc*>(static_cast<const Graph&>(*this).arc(i));
  }

  struct SharedGraph {
    /// Underlying graph data
    std::vector<Arc> arcs;
    std::vector<Node> nodes;
    std::vector<int> start;
    std::vector<int> accept;
    bool acceptor{true};
  };

  struct SharedGrad {
    /// Underlying grad data
    GradFunc gradFunc{nullptr};
    std::vector<Graph> inputs;
    std::unique_ptr<Graph> grad{nullptr};
    // TODO what are the implications here
    bool calcGrad;
  };


  std::shared_ptr<SharedGraph> sharedGraph_{std::make_shared<SharedGraph>()};
  std::shared_ptr<std::vector<float>> sharedWeights_{std::make_shared<std::vector<float>>()};
  std::shared_ptr<SharedGrad> sharedGrad_{std::make_shared<SharedGrad>()};
};

} // namespace gtn
