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
    // clears the gradient data in case it was set in `data`
    setCalcGrad(false);
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
  arcs().emplace_back(upNode, downNode, ilabel, olabel, weight, numArcs());
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

float Graph::item() const {
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

Graph& Graph::grad() {
  if (!calcGrad()) {
    throw std::logic_error("[Graph::grad] Gradient calculation disabled.");
  }
  if (!sharedData_->grad_) {
    throw std::logic_error("[Graph::grad] Gradient not calculated yet.");
  }
  return *sharedData_->grad_;
}

void Graph::addGrad(const Graph& other) {
  if (calcGrad()) {
    if (sharedData_->grad_) {
      auto& gradArcs = sharedData_->grad_->arcs();
      for (int i = 0; i < numArcs(); i++) {
        gradArcs[i].setWeight(
          gradArcs[i].weight() + other.arcs()[i].weight());
      }
    } else {
    // TODO Avoid the extra copy here if it's possible. Maybe we can have a "move" version
      sharedData_->grad_ = std::make_unique<Graph>(deepCopy(other));
    }
  }
}

void Graph::setCalcGrad(bool calcGrad) {
  sharedData_->calcGrad_ = calcGrad;
  if (!calcGrad) {
    sharedData_->gradFunc_ = nullptr;
    sharedData_->inputs_.clear();
    sharedData_->grad_.reset();
  }
}

void Graph::zeroGrad() {
  sharedData_->grad_.reset();
}

std::uintptr_t Graph::id() {
  return reinterpret_cast<std::uintptr_t>(sharedData_.get());
}

Graph Graph::deepCopy(const Graph& other) {
  Graph out(other.calcGrad());
  for (const auto& node : other.nodes()) {
    out.addNode(node.start(), node.accept());
  }
  for (const auto& arc : other.arcs()) {
    out.addArc(
        arc.upNode()->index(),
        arc.downNode()->index(),
        arc.ilabel(),
        arc.olabel(),
        arc.weight());
  }
  return out;
}

Node::Node(int index, bool start /* = false */, bool accept /* = false */)
    : index_(index), start_(start), accept_(accept) {}

void Node::addInArc(Arc* arc) {
  in_.push_back(arc);
}

void Node::addOutArc(Arc* arc) {
  out_.push_back(arc);
}

Arc::Arc(Node* upNode, Node* downNode, int ilabel, int olabel, float weight, int index)
    : upNode_(upNode),
      downNode_(downNode),
      ilabel_(ilabel),
      olabel_(olabel),
      weight_(weight),
      index_(index) {}
} // namespace gtn
