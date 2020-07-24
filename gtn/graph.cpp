#include <algorithm>
#include <stdexcept>

#include "graph.h"

namespace gtn {

Graph::Graph(GradFunc gradFunc, std::vector<Graph> inputs) {
  sharedGrad_->calcGrad = false;
  // If any inputs require a gradient, then this should
  // also compute a gradient.
  for (auto& g : inputs) {
    sharedGrad_->calcGrad |= g.calcGrad();
  }
  if (calcGrad()) {
    sharedGrad_->gradFunc = std::move(gradFunc);
    sharedGrad_->inputs = std::move(inputs);
  }
}

Graph::Graph(Graph& data, GradFunc gradFunc, std::vector<Graph> inputs)
    : sharedGraph_(data.sharedGraph_), sharedWeights_(data.sharedWeights_) {
  sharedGrad_->calcGrad = false;
  // If any inputs require a gradient, then this should
  // also compute a gradient.
  for (auto& g : inputs) {
    sharedGrad_->calcGrad |= g.calcGrad();
  }
  if (calcGrad()) {
    sharedGrad_->gradFunc = std::move(gradFunc);
    sharedGrad_->inputs = std::move(inputs);
  } else {
    // clears the gradient data in case it was set in `data`
    setCalcGrad(false);
  }
}

Graph::Graph(bool calcGrad /* = true */) {
  sharedGrad_->calcGrad = calcGrad;
}

int Graph::addNode(bool start /* = false */, bool accept /* = false */) {
  int idx = numNodes();
  sharedGraph_->nodes.emplace_back(start, accept);
  if (start) {
    sharedGraph_->start.push_back(idx);
  }
  if (accept) {
    sharedGraph_->accept.push_back(idx);
  }
  sharedGraph_->ilabelSorted = false;
  sharedGraph_->olabelSorted = false;
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
  assert(ilabel >= epsilon && olabel >= epsilon);
  auto idx = numArcs();
  sharedGraph_->arcs.emplace_back(upNode, downNode, ilabel, olabel);
  sharedWeights_->push_back(weight);
  node(upNode).out.push_back(idx);
  node(downNode).in.push_back(idx);
  sharedGraph_->ilabelSorted = false;
  sharedGraph_->olabelSorted = false;
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
  return const_cast<Graph&>(static_cast<const Graph&>(*this).grad());
}

const Graph& Graph::grad() const {
  if (!calcGrad()) {
    throw std::logic_error("[Graph::grad] Gradient calculation disabled.");
  }
  if (!sharedGrad_->grad) {
    throw std::logic_error("[Graph::grad] Gradient not calculated yet.");
  }
  return *sharedGrad_->grad;
}

void Graph::addGrad(std::vector<float>&& other) {
  if (calcGrad()) {
    if (other.size() != numArcs()) {
      throw std::logic_error("[Graph::addGrad] Invalid grad size.");
    }
    if (isGradAvailable()) {
      for (int i = 0; i < numArcs(); i++) {
        grad().setWeight(i, grad().weight(i) + other[i]);
      }
    } else {
      sharedGrad_->grad = std::make_unique<Graph>(false);
      sharedGrad_->grad->sharedGraph_ = sharedGraph_;
      *(sharedGrad_->grad->sharedWeights_) = std::move(other);
    }
  }
}

void Graph::addGrad(const std::vector<float>& other) {
  if (calcGrad()) {
    if (other.size() != numArcs()) {
      throw std::logic_error("[Graph::addGrad] Invalid grad size.");
    }
    if (isGradAvailable()) {
      for (int i = 0; i < numArcs(); i++) {
        grad().setWeight(i, grad().weight(i) + other[i]);
      }
    } else {
      // make a copy
      addGrad(std::vector<float>(other));
    }
  }
}

void Graph::addGrad(const Graph& other) {
  addGrad(*other.sharedWeights_);
}

void Graph::setCalcGrad(bool calcGrad) {
  sharedGrad_->calcGrad = calcGrad;
  if (!calcGrad) {
    sharedGrad_->gradFunc = nullptr;
    sharedGrad_->inputs.clear();
    sharedGrad_->grad.reset();
  }
}

void Graph::zeroGrad() {
  sharedGrad_->grad.reset();
}

std::uintptr_t Graph::id() {
  return reinterpret_cast<std::uintptr_t>(sharedGrad_.get());
}

Graph Graph::deepCopy(const Graph& src) {
  Graph out(src.calcGrad());
  out.sharedGraph_->arcs = src.sharedGraph_->arcs;
  out.sharedGraph_->nodes = src.sharedGraph_->nodes;
  out.sharedGraph_->start = src.sharedGraph_->start;
  out.sharedGraph_->accept = src.sharedGraph_->accept;
  *out.sharedWeights_ = *src.sharedWeights_;
  return out;
}

void Graph::arcSort(bool olabel /* = false */) {
  if ((olabel && sharedGraph_->olabelSorted) ||
      (!olabel && sharedGraph_->ilabelSorted)) {
    return;
  }
  sharedGraph_->olabelSorted = olabel;
  sharedGraph_->ilabelSorted = !olabel;
  auto sortFn = [olabel, &arcs = sharedGraph_->arcs](int a, int b) {
    return olabel ? arcs[a].olabel < arcs[b].olabel
                  : arcs[a].ilabel < arcs[b].ilabel;
  };
  for (auto& n : sharedGraph_->nodes) {
    std::sort(n.in.begin(), n.in.end(), sortFn);
    std::sort(n.out.begin(), n.out.end(), sortFn);
  }
}

void Graph::setWeights(const float* weights) {
  std::copy(weights, weights + numArcs(), sharedWeights_->data());
}

void Graph::labelsToArray(int* out, bool inLabel) {
  for (int i = 0; i < numArcs(); ++i) {
    out[i] = inLabel ? ilabel(i) : olabel(i);
  }
}

std::vector<int> Graph::labelsToVector(bool inLabel) {
  std::vector<int> out(numArcs());
  labelsToArray(out.data(), inLabel);
  return out;
}

} // namespace gtn
