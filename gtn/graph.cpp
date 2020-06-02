#include "graph.h"

namespace gtn {

Graph::Graph(GradFunc gradFunc, std::vector<Graph> inputs) {
  sharedData_->calcGrad_ = false;
  // If any inputs require a gradient, then this should
  // also compute a gradient.
  for (auto& g : inputs) {
    sharedData_->calcGrad_ |= g.calcGrad();
  }
  if (calcGrad()) {
    sharedData_->gradFunc_ = std::move(gradFunc);
    sharedData_->inputs_ = std::move(inputs);
  }
}

Graph::Graph(Graph& data, GradFunc gradFunc, std::vector<Graph> inputs)
    : sharedData_(data.sharedData_) {
  sharedData_->calcGrad_ = false;
  // If any inputs require a gradient, then this should
  // also compute a gradient.
  for (auto& g : inputs) {
    sharedData_->calcGrad_ |= g.calcGrad();
  }
  if (calcGrad()) {
    sharedData_->gradFunc_ = std::move(gradFunc);
    sharedData_->inputs_ = std::move(inputs);
  } else {
    // clear the gradFunc and inputs just in case they were set in `data`
    sharedData_->gradFunc_ = nullptr;
    sharedData_->inputs_ = {};
  }
}

Graph::Graph(bool calcGrad /* = true */) {
  sharedData_->calcGrad_ = calcGrad;
}

Node* Graph::addNode(bool start /* = false */, bool accept /* = false */) {
  nodes().emplace_back(numNodes(), start, accept);
  auto pNode = &nodes().back();
  if (start) {
    sharedData_->start_.push_back(pNode);
  }
  if (accept) {
    sharedData_->accept_.push_back(pNode);
  }
  return pNode;
}

Arc* Graph::addArc(Node* upNode, Node* downNode, int label) {
  return addArc(upNode, downNode, label, label);
}

Arc* Graph::addArc(
    Node* upNode,
    Node* downNode,
    int ilabel,
    int olabel,
    float weight /* = 0 */) {
  sharedData_->acceptor_ &= (ilabel == olabel);
  arcs().emplace_back(upNode, downNode, ilabel, olabel, weight);
  auto pArc = &arcs().back();
  upNode->addOutArc(pArc);
  downNode->addInArc(pArc);
  return pArc;
}

Arc* Graph::addArc(int upNode, int downNode, int label) {
  return addArc(upNode, downNode, label, label);
}

Arc* Graph::addArc(
    int upNode,
    int downNode,
    int ilabel,
    int olabel,
    float weight) {
  auto up = node(upNode);
  auto down = node(downNode);
  return addArc(up, down, ilabel, olabel, weight);
}

float Graph::item() {
  if (numArcs() != 1) {
    throw std::invalid_argument(
        "[Graph::item] Cannot convert Graph with more than 1 arc to a scalar.");
  }
  return arcs()[0].weight();
}

bool Graph::hasNode(int index) {
  return index >= 0 && index < numNodes();
}

Node* Graph::node(int index) {
  if (!hasNode(index)) {
    throw std::invalid_argument("[Graph::node] Node not in graph.");
  }
  return &nodes()[index];
}

void Graph::zeroGrad() {
  for (auto& arc : arcs()) {
    arc.zeroGrad();
  }
}

std::uintptr_t Graph::id() {
  return reinterpret_cast<std::uintptr_t>(sharedData_.get());
}

Node::Node(int index, bool start /* = false */, bool accept /* = false */)
    : index_(index), start_(start), accept_(accept) {}

void Node::addInArc(Arc* arc) {
  in_.push_back(arc);
}

void Node::addOutArc(Arc* arc) {
  out_.push_back(arc);
}

Arc::Arc(Node* upNode, Node* downNode, int ilabel, int olabel, float weight)
    : upNode_(upNode),
      downNode_(downNode),
      ilabel_(ilabel),
      olabel_(olabel),
      weight_(weight),
      grad_(0.0) {}
} // namespace gtn
