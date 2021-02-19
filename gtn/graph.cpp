/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <stdexcept>
#include <utility>

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

Graph::Graph(bool calcGrad /* = true */) {
  sharedGrad_->calcGrad = calcGrad;
}

int Graph::addNode(bool start /* = false */, bool accept /* = false */) {
  int idx = static_cast<int>(numNodes());
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

size_t Graph::addArc(size_t srcNode, size_t dstNode, int label) {
  return addArc(srcNode, dstNode, label, label);
}

size_t Graph::addArc(
    size_t srcNode,
    size_t dstNode,
    int ilabel,
    int olabel,
    float weight /* = 0 */) {
  assert(ilabel >= epsilon && olabel >= epsilon);
  int idx = static_cast<int>(numArcs());
  sharedGraph_->arcs.emplace_back(
      static_cast<int>(srcNode), static_cast<int>(dstNode), ilabel, olabel);
  sharedWeights_->push_back(weight);
  node(srcNode).out.push_back(idx);
  node(dstNode).in.push_back(idx);
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
    std::lock_guard<std::mutex> lock(sharedGraph_->grad_lock);
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
    std::lock_guard<std::mutex> lock(sharedGraph_->grad_lock);
    if (isGradAvailable()) {
      for (int i = 0; i < numArcs(); i++) {
        grad().setWeight(i, grad().weight(i) + other[i]);
      }
    } else {
      sharedGrad_->grad = std::make_unique<Graph>(false);
      sharedGrad_->grad->sharedGraph_ = sharedGraph_;
      *(sharedGrad_->grad->sharedWeights_) = other;
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

void Graph::setInputs(std::vector<Graph> inputs) {
  sharedGrad_->inputs = std::move(inputs);
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

void Graph::labelsToArray(int* out, bool ilabel) {
  for (int i = 0; i < numArcs(); ++i) {
    out[i] = ilabel ? this->ilabel(i) : olabel(i);
  }
}

std::vector<int> Graph::labelsToVector(bool ilabel) {
  std::vector<int> out(numArcs());
  labelsToArray(out.data(), ilabel);
  return out;
}

} // namespace gtn
