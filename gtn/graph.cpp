#include "graph.h"

namespace gtn {

Graph::Graph(GradFunc gradFunc, std::vector<Graph> inputs) {
  sharedData_->calcGrad = false;
  // If any inputs require a gradient, then this should
  // also compute a gradient.
  for (auto& g : inputs) {
    sharedData_->calcGrad |= g.calcGrad();
  }
  if (calcGrad()) {
    sharedData_->gradFunc = std::move(gradFunc);
    sharedData_->inputs = std::move(inputs);
  }
}

Graph::Graph(Graph& data, GradFunc gradFunc, std::vector<Graph> inputs)
    : sharedData_(data.sharedData_),
      arcs_(data.arcs_),
      nodes_(data.nodes_) {
  sharedData_->calcGrad = false;
  // If any inputs require a gradient, then this should
  // also compute a gradient.
  for (auto& g : inputs) {
    sharedData_->calcGrad |= g.calcGrad();
  }
  if (calcGrad()) {
    sharedData_->gradFunc = std::move(gradFunc);
    sharedData_->inputs = std::move(inputs);
  } else {
    // clears the gradient data in case it was set in `data`
    setCalcGrad(false);
  }
}

Graph::Graph(bool calcGrad /* = true */) {
  sharedData_->calcGrad = calcGrad;
}

int Graph::addNode(bool start /* = false */, bool accept /* = false */) {
  int idx = numNodes();
  sharedData_->nodes.emplace_back(start, accept);
  if (start) {
    sharedData_->start.push_back(idx);
  }
  if (accept) {
    sharedData_->accept.push_back(idx);
  }
  nodes_ = sharedData_->nodes.data();
  return idx;
}

int Graph::addArc(int upNode, int downNode, int label) {
  return addArc(upNode, downNode, label, label);
}

int Graph::addArc(
    int upNode,
    int downNode,
    int ilabel,
    int olabel,
    float weight /* = 0 */) {
  sharedData_->acceptor &= (ilabel == olabel);
  auto idx = numArcs();
  sharedData_->arcs.emplace_back(upNode, downNode, ilabel, olabel, weight);
  arcs_ = sharedData_->arcs.data();
  (nodes_ + upNode)->out.push_back(idx);
  (nodes_ + downNode)->in.push_back(idx);
  return idx;
}

float Graph::item() const {
  if (numArcs() != 1) {
    throw std::invalid_argument(
        "[Graph::item] Cannot convert Graph with more than 1 arc to a scalar.");
  }
  return weight(0);
}

Graph& Graph::grad() {
  if (!calcGrad()) {
    throw std::logic_error("[Graph::grad] Gradient calculation disabled.");
  }
  if (!sharedData_->grad) {
    throw std::logic_error("[Graph::grad] Gradient not calculated yet.");
  }
  return *sharedData_->grad;
}

void Graph::addGrad(Graph&& other) {
  if (calcGrad()) {
    if (isGradAvailable()) {
      for (int i = 0; i < numArcs(); i++) {
        grad().setWeight(i, grad().weight(i) + other.weight(i));
      }
    } else {
      sharedData_->grad = std::make_unique<Graph>(other);
    }
  }
}

void Graph::addGrad(const Graph& other) {
  if (calcGrad()) {
    if (sharedData_->grad) {
      // NB: this is safe because we don't keep a reference
      // to other if the grad exists
      addGrad(std::move(const_cast<Graph&>(other)));
    } else {
      addGrad(deepCopy(other));
    }
  }
}

void Graph::setCalcGrad(bool calcGrad) {
  sharedData_->calcGrad = calcGrad;
  if (!calcGrad) {
    sharedData_->gradFunc = nullptr;
    sharedData_->inputs.clear();
    sharedData_->grad.reset();
  }
}

void Graph::zeroGrad() {
  sharedData_->grad.reset();
}

std::uintptr_t Graph::id() {
  return reinterpret_cast<std::uintptr_t>(sharedData_.get());
}

Graph Graph::deepCopy(const Graph& src) {
  Graph out(src.calcGrad());
  out.sharedData_->arcs = src.sharedData_->arcs;
  out.sharedData_->nodes = src.sharedData_->nodes;
  out.sharedData_->start = src.sharedData_->start;
  out.sharedData_->accept = src.sharedData_->accept;
  out.sharedData_->acceptor = src.sharedData_->acceptor;
  out.arcs_ = out.sharedData_->arcs.data();
  out.nodes_ = out.sharedData_->nodes.data();
  return out;
}

} // namespace gtn
