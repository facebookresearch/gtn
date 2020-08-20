#pragma once

#include <cassert>
#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <mutex>

namespace gtn {

/* Graph operations are in the log or tropical semirings. The default score
 * for an arc is 0 (e.g. the multiplicative identity) and the additive
 * identity is -infinity. Path scores are accumulated with the logadd or
 * max operations and the score for a path is accumulated with addition. */

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
    Arc(int upNode, int downNode, int ilabel, int olabel)
        : upNode(upNode), downNode(downNode), ilabel(ilabel), olabel(olabel){};
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
   * @param upNode The id of the source node.
   * @param downNode The id of the destination node.
   * @param label The arc label.
   * @return The id of the added arc.
   */
  int addArc(int upNode, int downNode, int label);

  /**
   * Add a arc between two nodes.
   * @param upNode The id of the source node.
   * @param downNode The id of the destination node.
   * @param ilabel The arc input label.
   * @param olabel The arc output label.
   * @param weight The arc weight.
   * @return The id of the added arc.
   */
  int addArc(
      int upNode,
      int downNode,
      int ilabel,
      int olabel,
      float weight = 0.0);

  // Attempt to keep code like `g.addArc(n1, n2, 0, 2.0)` from compiling
  int addArc(int upNode, int downNode, int label, float) = delete;
  int addArc(int upNode, int downNode, int label, double) = delete;

  /** The number of arcs in the graph. */
  int numArcs() const {
    return sharedGraph_->arcs.size();
  };
  /** The number of nodes in the graph. */
  int numNodes() const {
    return sharedGraph_->nodes.size();
  };
  /** The number of starting nodes in the graph. */
  int numStart() const {
    return sharedGraph_->start.size();
  };
  /** The number of accepting nodes in the graph. */
  int numAccept() const {
    return sharedGraph_->accept.size();
  };

  /** Get the weight on a single arc graph.  */
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
  const Graph& grad() const;

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

  /* Clear the weights on a graph if they are no longer needed. */
  Graph withoutWeights() const {
    Graph other = *this;
    other.sharedWeights_ = nullptr;
    return other;
  }

  static constexpr int epsilon{-1};

  // Accessing and modifying nodes.

  const std::vector<int>& start() const {
    return sharedGraph_->start;
  };
  const std::vector<int>& accept() const {
    return sharedGraph_->accept;
  };
  bool start(int i) const {
    return node(i).start;
  };
  bool accept(int i) const {
    return node(i).accept;
  };
  void makeAccept(int i) {
    auto& n = node(i);
    if (!n.accept) {
      sharedGraph_->accept.push_back(i);
      n.accept = true;
    }
  };
  int numOut(int i) const {
    return node(i).out.size();
  }
  const std::vector<int>& out(int i) const {
    return node(i).out;
  }
  int out(int i, int j) const {
    return node(i).out[j];
  }
  int numIn(int i) const {
    return node(i).in.size();
  }
  const std::vector<int>& in(int i) const {
    return node(i).in;
  }
  int in(int i, int j) const {
    return node(i).in[j];
  }

  // Accessing and modifying arcs.

  int upNode(int i) const {
    return arc(i).upNode;
  }
  int downNode(int i) const {
    return arc(i).downNode;
  }
  int label(int i) const {
    return arc(i).ilabel;
  }
  int ilabel(int i) const {
    return arc(i).ilabel;
  }
  int olabel(int i) const {
    return arc(i).olabel;
  }
  float weight(int i) const {
    assert(sharedWeights_ != nullptr);
    return (*sharedWeights_)[i];
  }
  void setWeight(int i, float weight) {
    assert(sharedWeights_ != nullptr);
    (*sharedWeights_)[i] = weight;
  }

  /* Sort the arcs entering and exiting a node in increasing order by arc in
   * label (default) or out label if `olabel = True`. This function is intended
   * to be used prior to calls to `intersect` and `compose` to improve the
   * efficiency of the algorithm. */
  void arcSort(bool olabel = false);

  /* Mark a graph's arcs as sorted.
   * If `olabel == false` (default) then the graph will be marked as sorted by
   * arc input labels, otherwise it will be marked as sorted by the arc output
   * labels. */
  void markArcSorted(bool olabel = false) {
    if (olabel) {
      sharedGraph_->olabelSorted = true;
    } else {
      sharedGraph_->ilabelSorted = true;
    }
  }

  /* Check if the arcs entering and exiting a node are sorted by in label. */
  bool ilabelSorted() const {
    return sharedGraph_->ilabelSorted;
  }

  /* Check if the arcs entering and exiting a node are sorted by out label. */
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
   * Extract a vector of labels from the graph. See `Graph::labelsToArray`.
   */
  std::vector<int> labelsToVector(bool ilabel = true);

 private:
  const Node& node(int i) const {
    // NB: assert gets stripped at in release mode
    assert(i >= 0 && i < numNodes());
    return sharedGraph_->nodes[i];
  }
  Node& node(int i) {
    return const_cast<Node&>(static_cast<const Graph&>(*this).node(i));
  }
  const Arc& arc(int i) const {
    // NB: assert gets stripped at in release mode
    assert(i >= 0 && i < numArcs());
    return sharedGraph_->arcs[i];
  }
  Arc& arc(int i) {
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
