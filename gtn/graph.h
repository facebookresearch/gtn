/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace gtn {

/** The index of the epsilon label. */
constexpr int epsilon{-1};

/**
 * A `Graph` class to perform automatic differentiation with weighted
 * finite-state acceptors (WFSAs) and transducers (WFSTs).
 *
 * Example:
 *
 * \code{.cpp}
 * Graph graph;
 * graph.addNode(true); // Add a start node
 * graph.addNode(); // Add an internal node
 * graph.addNode(false, true); // Add an accept node
 *
 * // Add an arc from node 0 to 1 with ilabel 0, olabel 1 and weight 2.0
 * graph.addArc(0, 1, 0, 1, 2.0);
 *
 * // Add an arc from node 1 to 2 with ilabel 1, olabel 2 and weight 1.0
 * graph.addArc(1, 2, 1, 2, 1.0);
 *
 * // Compute the Viterbi score of the graph
 * auto score = viterbiScore(graph);
 *
 * print(score); // Print the score graph to std out
 *
 * backward(score); // Compute the gradient
 * graph.grad(); // Access the gradient
 * graph.zeroGrad(); // Clear the gradient
 * \endcode
 *
 * All operations are in the log or tropical semirings. The default score
 * for an arc is `0` (e.g. the multiplicative identity) and the additive
 * identity is `-infinity`. Path scores are accumulated with log-sum-exp or
 * max operations and the score for a path is accumulated with addition.
 */
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
    Arc(int srcNode, int dstNode, int ilabel, int olabel)
        : srcNode(srcNode), dstNode(dstNode), ilabel(ilabel), olabel(olabel){};
    int srcNode;
    int dstNode;
    int ilabel;
    int olabel;
  };

 public:
  using GradFunc =
      std::function<void(std::vector<Graph>& inputs, Graph& deltas)>;
  Graph(GradFunc gradFunc, std::vector<Graph> inputs);

  /**
   * \defgroup graphMethods Graph-level methods
   * @{
   */

  /** Construct a `Graph`.
   * @param calcGrad Whether or not to compute gradients with respect to this
   *   graph when calling `gtn::backward`.
   */
  Graph(bool calcGrad = true);

  /**
   * Adds a node to the graph.
   * @param start Indicates if the node is a starting node.
   * @param accept Indicates if the node is an accepting node.
   * @return The id of the node (used for e.g. adding arcs).
   */
  int addNode(bool start = false, bool accept = false);

  /**
   * Add a arc between two nodes. This assumes the graph is an acceptor, the
   * input label on the arc is the same as the output label.
   * @param srcNode The id of the source node.
   * @param dstNode The id of the destination node.
   * @param label The arc label.
   * @return The id of the added arc.
   */
  size_t addArc(size_t srcNode, size_t dstNode, int label);

  /**
   * Add a arc between two nodes.
   * @param srcNode The id of the source node.
   * @param dstNode The id of the destination node.
   * @param ilabel The arc input label.
   * @param olabel The arc output label.
   * @param weight The arc weight.
   * @return The id of the added arc.
   */
  size_t addArc(
      size_t srcNode,
      size_t dstNode,
      int ilabel,
      int olabel,
      float weight = 0.0);

  /** The number of arcs in the graph. */
  size_t numArcs() const {
    return sharedGraph_->arcs.size();
  };
  /** The number of nodes in the graph. */
  size_t numNodes() const {
    return sharedGraph_->nodes.size();
  };
  /** The number of starting nodes in the graph. */
  size_t numStart() const {
    return sharedGraph_->start.size();
  };
  /** The number of accepting nodes in the graph. */
  size_t numAccept() const {
    return sharedGraph_->accept.size();
  };

  /** Get the weight on a single arc graph.  */
  float item() const;

  /**
   * A deep copy of a graph `src` which is not recorded in the
   * autograd tape. For a version which is recorded in the
   * autograd tape see `gtn::clone`.
   */
  static Graph deepCopy(const Graph& src);

  /**
   * Sort the arcs entering and exiting a node in increasing order by arc in
   * label or out label if `olabel == true`. This function is intended
   * to be used prior to calls to `intersect` and `compose` to improve the
   * efficiency of the algorithm.
   */
  void arcSort(bool olabel = false);

  /**
   * Mark a graph's arcs as sorted.
   * If `olabel == false` then the graph will be marked as sorted by
   * arc input labels, otherwise it will be marked as sorted by the arc output
   * labels.
   */
  void markArcSorted(bool olabel = false) {
    if (olabel) {
      sharedGraph_->olabelSorted = true;
    } else {
      sharedGraph_->ilabelSorted = true;
    }
  }

  /**
   * Check if the arcs entering and exiting every node are sorted by input
   * label.
   */
  bool ilabelSorted() const {
    return sharedGraph_->ilabelSorted;
  }

  /**
   * Check if the arcs entering and exiting every node are sorted by output
   * label.
   */
  bool olabelSorted() const {
    return sharedGraph_->olabelSorted;
  }

  /**
   * Returns an array of weights from a graph. The array will contain
   * `Graph::numArcs()` elements.
   */
  float* weights() {
    assert(sharedWeights_ != nullptr);
    return sharedWeights_->data();
  }
  /**
   * A `const` version of `Graph::weights`.
   */
  const float* weights() const {
    assert(sharedWeights_ != nullptr);
    return sharedWeights_->data();
  }

  /**
   * Set the arc weights on a graph. The `weights` array must have
   * `Graph::numArcs()` elements.
   */
  void setWeights(const float* weights);

  /**
   * Extract an array of labels from a graph. The array should have space for
   * `Graph::numArcs()` elements.
   *
   * @param[out] out A pointer to the buffer to populate with labels.
   * @param[in] ilabel Retreive ilabels if true, otherwise gets olabels.
   */
  void labelsToArray(int* out, bool ilabel = true);

  /**
   * Extract a `std::vector` of labels from the graph. See
   * `Graph::labelsToArray`.
   */
  std::vector<int> labelsToVector(bool ilabel = true);

  /** @}*/

  /**
   * \defgroup gradMethods Autograd methods
   * @{
   */

  /**
   * Add a `std::vector` of gradients to the gradient graph weights without
   * making a copy of `other`. The `Graph::addGrad` methods are intended for
   * use by the autograd.
   * This overload is used with an `rvalue` or `std::move` to avoid an extra
   * copy:
   * \code{.cpp}
   * graph.addGrad(std::move(graphGrad));
   * \endcode
   */
  void addGrad(std::vector<float>&& other);

  /**
   * Add a `std::vector` of gradients to the gradient graph weights. The
   * `Graph::addGrad` methods are intended for use by the autograd.
   */
  void addGrad(const std::vector<float>& other);

  /**
   * Add a `Graph` of gradients to the gradient graph. The `Graph::addGrad`
   * methods are intended for use by the autograd.
   */
  void addGrad(const Graph& other);

  /** Check if a graph requires a gradient. */
  bool calcGrad() const {
    return sharedGrad_->calcGrad;
  };
  /** Check if a graph's gradient is computed. */
  bool isGradAvailable() const {
    return sharedGrad_->grad != nullptr;
  }
  /** Get the gradient graph. */
  Graph& grad();

  /** A `const` version of `Graph::grad`. */
  const Graph& grad() const;

  /** Specify if the gradient for this graph should be computed. */
  void setCalcGrad(bool calcGrad);

  /** Clear the graph's gradients. */
  void zeroGrad();

  /**
   * A unique identifier for a graph. Intended for use by the autograd.
   */
  std::uintptr_t id();

  /**
   * Get the gradient function of a graph. Intended for use by the autograd.
   */
  GradFunc gradFunc() {
    return sharedGrad_->gradFunc;
  };

  /**
   * Set the gradient function of a graph. Intended for use by the autograd.
   */
  void setGradFunc(GradFunc gradFunc) {
    if (calcGrad()) {
      sharedGrad_->gradFunc = gradFunc;
    }
  }

  /**
   * Get the vector of inputs used in the autograd computation graph. Intended
   * for use by the autograd.
   */
  std::vector<Graph>& inputs() const {
    return sharedGrad_->inputs;
  };

  /**
   * Sets the vector of inputs used in the autograd computation graph. Intended
   * for use by the autograd.
   */
  void setInputs(std::vector<Graph> inputs);

  /**
   * Clear the weights on a graph if they are no longer needed. Intended for
   * use by the autograd.
   */
  Graph withoutWeights() const {
    Graph other = *this;
    other.sharedWeights_ = nullptr;
    return other;
  }

  /** @} */

  /** \defgroup nodeAccess Node accessors
   *  @{
   */

  /** Get the indices of the start nodes of the graph. */
  const std::vector<int>& start() const {
    return sharedGraph_->start;
  };
  /** Get the indices of the accepting nodes of the graph. */
  const std::vector<int>& accept() const {
    return sharedGraph_->accept;
  };
  /** Check if the `i`-th node is a start node. */
  bool isStart(size_t i) const {
    return node(i).start;
  };
  /** Check if the `i`-th node is an accepting node. */
  bool isAccept(size_t i) const {
    return node(i).accept;
  };
  /** Make the the `i`-th node an accepting node. */
  void makeAccept(size_t i) {
    auto& n = node(i);
    if (!n.accept) {
      sharedGraph_->accept.push_back(static_cast<int>(i));
      n.accept = true;
    }
  };
  /** The number of outgoing arcs from the `i`-th node. */
  size_t numOut(size_t i) const {
    return node(i).out.size();
  }
  /** Get the indices of outgoing arcs from the `i`-th node. */
  const std::vector<int>& out(size_t i) const {
    return node(i).out;
  }
  /** Get the index of the `j`-th outgoing arc from the `i`-th node. */
  int out(size_t i, size_t j) const {
    return node(i).out[j];
  }
  /** The number of incoming arcs to the `i`-th node. */
  size_t numIn(size_t i) const {
    return node(i).in.size();
  }
  /** Get the indices of incoming arcs to the `i`-th node. */
  const std::vector<int>& in(size_t i) const {
    return node(i).in;
  }
  /** Get the index of the `j`-th incoming arc to the `i`-th node. */
  size_t in(size_t i, size_t j) const {
    return node(i).in[j];
  }

  /** @}*/

  /** \defgroup arcAccess Arc accessors
   *  @{
   */

  /** The destination node of the `i`-th arc. */
  int srcNode(size_t i) const {
    return arc(i).srcNode;
  }
  /** The source node of the `i`-th arc. */
  int dstNode(size_t i) const {
    return arc(i).dstNode;
  }
  /** The label of the `i`-th arc (use this for acceptors). */
  int label(size_t i) const {
    return arc(i).ilabel;
  }
  /** The input label of the `i`-th arc. */
  int ilabel(size_t i) const {
    return arc(i).ilabel;
  }
  /** The output label of the `i`-th arc. */
  int olabel(size_t i) const {
    return arc(i).olabel;
  }

  /** The weight of the `i`-th arc. */
  float weight(size_t i) const {
    assert(sharedWeights_ != nullptr);
    return (*sharedWeights_)[i];
  }
  /** Set the weight of the `i`-th arc. */
  void setWeight(size_t i, float weight) {
    assert(sharedWeights_ != nullptr);
    (*sharedWeights_)[i] = weight;
  }
  /** @}*/

 private:
  // Attempt to keep code like `g.addArc(n1, n2, 0, 2.0)` from compiling
  size_t addArc(size_t srcNode, size_t dstNode, int label, float) = delete;
  size_t addArc(size_t srcNode, size_t dstNode, int label, double) = delete;

  const Node& node(size_t i) const {
    // NB: assert gets stripped at in release mode
    assert(i >= 0 && i < numNodes());
    return sharedGraph_->nodes[i];
  }
  Node& node(size_t i) {
    return const_cast<Node&>(static_cast<const Graph&>(*this).node(i));
  }
  const Arc& arc(size_t i) const {
    // NB: assert gets stripped at in release mode
    assert(i >= 0 && i < numArcs());
    return sharedGraph_->arcs[i];
  }
  Arc& arc(size_t i) {
    return const_cast<Arc&>(static_cast<const Graph&>(*this).arc(i));
  }

  struct SharedGraph {
    /// Underlying graph data
    std::vector<Arc> arcs;
    std::vector<Node> nodes;
    std::vector<int> start;
    std::vector<int> accept;

    // Some optional metadata about the graph
    bool ilabelSorted{false};
    bool olabelSorted{false};

    std::mutex grad_lock;
  };

  struct SharedGrad {
    /// Underlying grad data
    GradFunc gradFunc{nullptr};
    std::vector<Graph> inputs;
    std::unique_ptr<Graph> grad{nullptr};
    bool calcGrad;
  };

  std::shared_ptr<SharedGraph> sharedGraph_{std::make_shared<SharedGraph>()};
  std::shared_ptr<std::vector<float>> sharedWeights_{
      std::make_shared<std::vector<float>>()};
  std::shared_ptr<SharedGrad> sharedGrad_{std::make_shared<SharedGrad>()};
};

} // namespace gtn
