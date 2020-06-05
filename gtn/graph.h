#pragma once

#include <climits>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <vector>
#include <iostream>

namespace gtn {

// Forward declare.
class Node;

class Arc {
 public:
  /* Graph operations are in the log or tropical semirings. The default score
   * for an arc is 0 (e.g. the multiplicative identity) and the additive
   * identity is -infinity. Path scores are accumulated with the logadd or
   * max operations and the score for a path is accumulated with addition. */
  Arc(Node* upNode, Node* downNode, int ilabel, int olabel, float weight);

  Node* upNode() const {
    return upNode_;
  };
  Node* downNode() const {
    return downNode_;
  };
  int ilabel() const {
    return ilabel_;
  };
  int olabel() const {
    return olabel_;
  };
  int label() const {
    return ilabel_;
  };
  float weight() const {
    return weight_;
  };
  void setWeight(float weight) {
    weight_ = weight;
  };

  // Autograd functionality
  float grad() const {
    return grad_;
  };
  void addGrad(float grad) {
    grad_ += grad;
  };
  void zeroGrad() {
    grad_ = 0;
  };

 private:
  Node* upNode_;
  Node* downNode_;
  int ilabel_;
  int olabel_;
  float weight_;
  float grad_;
};

class Node {
 public:
  Node(int index, bool start = false, bool accept = false);
  void addInArc(Arc* arc);
  void addOutArc(Arc* arc);
  int numIn() const {
    return in_.size();
  };
  int numOut() const {
    return out_.size();
  };
  int index() const {
    return index_;
  };
  bool start() const {
    return start_;
  };
  bool accept() const {
    return accept_;
  };
  const std::vector<Arc*>& in() const {
    return in_;
  };
  const std::vector<Arc*>& out() const {
    return out_;
  };
  void setAccept(bool accept) {
    accept_ = accept;
  };

 private:
  int index_;
  bool start_;
  bool accept_;
  std::vector<Arc*> in_;
  std::vector<Arc*> out_;
};

class Graph {
 public:
  using GradFunc =
      std::function<void(std::vector<Graph>& inputs, Graph& deltas)>;

  Graph(GradFunc gradFunc, std::vector<Graph> inputs);
  Graph(Graph& data, GradFunc gradFunc, std::vector<Graph> inputs);
  Graph(bool calcGrad = true);

  Node* addNode(bool start = false, bool accept = false);

  /* Add an arc between two nodes. */
  Arc* addArc(Node* upNode, Node* downNode, int label);
  Arc* addArc(
      Node* upNode,
      Node* downNode,
      int ilabel,
      int olabel,
      float weight = 0.0);

  /* Add an arc between two nodes given the node indices.
   * Do not use this version for performance sensitive
   * code. */
  Arc* addArc(int upNode, int downNode, int label);
  Arc*
  addArc(int upNode, int downNode, int ilabel, int olabel, float weight = 0.0);

  // Attempt to keep code like `g.addArc(n1, n2, 0, 2.0)` from compiling
  Arc* addArc(Node* upNode, Node* downNode, int label, float) = delete;
  Arc* addArc(Node* upNode, Node* downNode, int label, double) = delete;
  Arc* addArc(int upNode, int downNode, int label, float) = delete;
  Arc* addArc(int upNode, int downNode, int label, double) = delete;

  // Get the score on a single arc graph.
  float item();

  bool hasNode(int index);
  Node* node(int index);

  std::deque<Arc>& arcs() {
    return sharedData_->arcs_;
  };
  std::deque<Node>& nodes() {
    return sharedData_->nodes_;
  };
  const std::vector<Node*>& start() const {
    return sharedData_->start_;
  };
  const std::vector<Node*>& accept() const {
    return sharedData_->accept_;
  };
  int numArcs() const {
    return sharedData_->arcs_.size();
  };
  int numNodes() const {
    return sharedData_->nodes_.size();
  };
  int numStart() const {
    return sharedData_->start_.size();
  };
  int numAccept() const {
    return sharedData_->accept_.size();
  };
  bool acceptor() const {
    return sharedData_->acceptor_;
  }
  bool calcGrad() const {
    return sharedData_->calcGrad_;
  };

  void makeAccept(Node* n) {
    if (!n->accept()) {
      sharedData_->accept_.push_back(n);
      n->setAccept(true);
    }
  };

  void zeroGrad();
  std::uintptr_t id();
  GradFunc gradFunc() {
    return sharedData_->gradFunc_;
  };
  std::vector<Graph> inputs() {
    return sharedData_->inputs_;
  };

  static constexpr int epsilon{-1};

 private:
  struct SharedData {
    /// Underlying graph data
    // arcs_ is the sole owner of Arcs in the graph.
    std::deque<Arc> arcs_;
    // nodes_ is the sole owner of Nodes in the graph
    std::deque<Node> nodes_;
    std::vector<Node*> start_;
    std::vector<Node*> accept_;
    GradFunc gradFunc_{nullptr};
    std::vector<Graph> inputs_;
    bool acceptor_{true};
    bool calcGrad_;
  };

  std::shared_ptr<SharedData> sharedData_{std::make_shared<SharedData>()};
};
} // namespace gtn
