#pragma once

#include <cassert>
#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace gtn {

class Graph {
 private:
  struct Node {
    Node(bool start, bool accept) : start(start), accept(accept){};
    bool start{false};
    bool accept{false};
    std::vector<int> in;
    std::vector<int> out;
  };

  struct Arc {
    /* Graph operations are in the log or tropical semirings. The default score
     * for an arc is 0 (e.g. the multiplicative identity) and the additive
     * identity is -infinity. Path scores are accumulated with the logadd or
     * max operations and the score for a path is accumulated with addition. */
    Arc(int upNode, int downNode, int ilabel, int olabel, float weight)
        : upNode(upNode),
          downNode(downNode),
          ilabel(ilabel),
          olabel(olabel),
          weight(weight){};
    int upNode;
    int downNode;
    int ilabel;
    int olabel;
    float weight;
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
    return sharedData_->arcs.size();
  };
  int numNodes() const {
    return sharedData_->nodes.size();
  };
  int numStart() const {
    return sharedData_->start.size();
  };
  int numAccept() const {
    return sharedData_->accept.size();
  };
  bool acceptor() const {
    return sharedData_->acceptor;
  }

  /* Get the score on a single arc graph. */
  float item() const;

  void addGrad(const Graph& other);
  void addGrad(Graph&& other);

  bool calcGrad() const {
    return sharedData_->calcGrad;
  };
  bool isGradAvailable() const {
    return sharedData_->grad != nullptr;
  }

  Graph& grad();

  void setCalcGrad(bool calcGrad);
  void zeroGrad();
  std::uintptr_t id();
  GradFunc gradFunc() {
    return sharedData_->gradFunc;
  };
  std::vector<Graph>& inputs() {
    return sharedData_->inputs;
  };

  /* A deep copy of a graph `other` which is not recorded in the
   * autograd tape. For a version which is recorded in the
   * autograd tape see `clone`. */
  static Graph deepCopy(const Graph& src);

  static constexpr int epsilon{-1};

  // Accessing and modifying nodes.

  const std::vector<int>& start() const {
    return sharedData_->start;
  };
  const std::vector<int>& accept() const {
    return sharedData_->accept;
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
      sharedData_->accept.push_back(i);
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
    return arc(i)->weight;
  }
  void setWeight(int i, float weight) {
    arc(i)->weight = weight;
  }

 private:
  const Node* node(int i) const {
    // NB: assert gets stripped at in release mode
    assert(i >= 0 && i < numNodes());
    return (nodes_ + i);
  }
  Node* node(int i) {
    return const_cast<Node*>(static_cast<const Graph&>(*this).node(i));
  }
  const Arc* arc(int i) const {
    // NB: assert gets stripped at in release mode
    assert(i >= 0 && i < numArcs());
    return (arcs_ + i);
  }
  Arc* arc(int i) {
    return const_cast<Arc*>(static_cast<const Graph&>(*this).arc(i));
  }

  struct SharedData {
    /// Underlying graph data
    // arcs is the sole owner of Arcs in the graph.
    std::vector<Arc> arcs;
    // nodes is the sole owner of Nodes in the graph
    std::vector<Node> nodes;
    std::vector<int> start;
    std::vector<int> accept;
    GradFunc gradFunc{nullptr};
    std::vector<Graph> inputs;
    bool acceptor{true};
    std::unique_ptr<Graph> grad{nullptr};
    bool calcGrad;
  };

  std::shared_ptr<SharedData> sharedData_{std::make_shared<SharedData>()};
  Arc *arcs_ = nullptr;
  Node *nodes_ = nullptr;
};

} // namespace gtn
