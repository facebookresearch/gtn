
#include <cassert>
#include <sstream>

#include "gtn/gtn.h"

using namespace gtn;

std::unordered_map<int, std::string> symbols = {{0, "a"}, {1, "b"}, {2, "c"}};
auto isymbols = symbols;
std::unordered_map<int, std::string> osymbols = {{0, "x"}, {1, "y"}, {2, "z"}};

// Build graph and simple utilities.
void simpleAcceptors() {
  // Create a graph, by default a weighted finite state acceptor (WFSA)
  Graph graph;
  // Add start node
  graph.addNode(true);

  // Add an accept node
  graph.addNode(false, true);

  // Add an internal node
  graph.addNode();

  // Add an arc from node 0 to 2 with label 0
  graph.addArc(0, 2, 0);

  // Add an arc from node 0 to 2 with input label 1 and output label 1
  graph.addArc(0, 2, 1, 1);

  // Add an arc from node 2 to 1 with input label 0, output label 0 and weight 2
  graph.addArc(2, 1, 0, 0, 2);

  // Print graph to std::out
  print(graph);

  // Draw the graph in dot format to file
  draw(graph, "simple_fsa.dot", symbols);
  // Compile to pdf with
  // dot -Tpdf graph.dot -o graph.pdf

  // One can load a graph from an istream
  std::stringstream in(
      // First line is space separated  start states
      "0\n"
      // Second line is space separated accept states
      "1\n"
      // The remaining lines are a list of arcs:
      // <source node> <dest node> <ilabel> [olabel] [weight]
      // where the olabel defaults to the ilabel if it is not specified
      // and the weight defaults to 0.0 if it is not specified.
      "0 2 0\n" // olabel = 0, weight = 0.0
      "0 2 1 1\n" // olabel = 1, weight = 0.0
      "2 1 0 0 2\n"); // olabel = 0, weight = 2.0

  Graph other_graph = load(in);

  // Exact match the two graphs, the node indices,
  // arc weights and arc labels should all be identical.
  assert(equals(graph, other_graph));

  // Check that the graphs have the same structure but
  // potentially different state indices. In this case,
  // only the arc labels and weights must be the same.
  assert(isomorphic(graph, other_graph));
}

// A few more interesting graphs
void interestingAcceptors() {
  {
    Graph graph;
    graph.addNode(true);
    // Graphs can have multiple start-nodes
    graph.addNode(true);
    graph.addNode();
    graph.addNode(false, true);
    // Graphs can also have multiple accept nodes
    graph.addNode(false, true);

    // Start nodes can have incoming arcs
    graph.addArc(0, 1, 1);
    graph.addArc(0, 2, 0);
    graph.addArc(1, 3, 0);
    graph.addArc(2, 3, 1);
    graph.addArc(2, 3, 0);
    graph.addArc(2, 4, 2);
    // Accept nodes can have outgoing arcs
    graph.addArc(3, 4, 1);

    draw(graph, "multi_start_accept.dot", symbols);
  }

  {
    Graph graph;
    graph.addNode(true);
    graph.addNode();
    graph.addNode(false, true);

    // Self loops are allowed
    graph.addArc(0, 0, 0);
    graph.addArc(0, 1, 1);
    graph.addArc(0, 1, 2);
    graph.addArc(1, 2, 1);
    // Cycles are also allowed
    graph.addArc(2, 0, 1);

    draw(graph, "cycles.dot", symbols);
  }

  {
    // Epsilon transitions
    Graph graph;
    graph.addNode(true);
    graph.addNode();
    graph.addNode(false, true);

    graph.addArc(0, 1, 0);
    graph.addArc(0, 1, Graph::epsilon);
    graph.addArc(1, 2, 1);
    draw(graph, "epsilons.dot", symbols);
  }
}

// Simple operations on WFSAs (and WFSTs)
void simpleOps() {
  // The sum (union) of a set of graphs accepts any sequence accepted by any
  // input graph.
  {
    // Recognizes "aba*"
    Graph g1;
    g1.addNode(true);
    g1.addNode();
    g1.addNode(false, true);
    g1.addArc(0, 1, 0);
    g1.addArc(1, 2, 1);
    g1.addArc(2, 2, 0);

    // Recognizes "ba"
    Graph g2;
    g2.addNode(true);
    g2.addNode();
    g2.addNode(false, true);
    g2.addArc(0, 1, 1);
    g2.addArc(1, 2, 0);

    // Recognizes "ac"
    Graph g3;
    g3.addNode(true);
    g3.addNode();
    g3.addNode(false, true);
    g3.addArc(0, 1, 0);
    g3.addArc(1, 2, 2);

    draw(g1, "sum_g1.dot", symbols);
    draw(g2, "sum_g2.dot", symbols);
    draw(g3, "sum_g3.dot", symbols);
    auto graph = sum({g1, g2, g3});
    draw(graph, "sum_graph.dot", symbols);
  }

  // The closure of a graph accepts any sequence accepted by the original graph
  // repeated 0 or more times (0 repeats is the empty sequence
  // "Graph::epsilon").
  {
    // Recognizes "aba"
    Graph g;
    g.addNode(true);
    g.addNode();
    g.addNode();
    g.addNode(false, true);
    g.addArc(0, 1, 0);
    g.addArc(1, 2, 1);
    g.addArc(2, 3, 0);
    draw(g, "closure_input.dot", symbols);

    auto graph = closure(g);
    draw(graph, "closure_graph.dot", symbols);
  }
}

// Composing WFSAs
void composingAcceptors() {
  // The composition of two acceptors is the graph which represents the set of
  // all paths present in both. More precisely, this would be called the
  // intersection, though the algorithm is the same. The score for a path in
  // the composed graph should be the sum of the scores for the path in the two
  // input graphs.
  Graph g1;
  g1.addNode(true);
  g1.addNode(false, true);
  g1.addArc(0, 0, 0);
  g1.addArc(0, 1, 1);
  g1.addArc(1, 1, 2);

  Graph g2;
  g2.addNode(true);
  g2.addNode();
  g2.addNode();
  g2.addNode(false, true);
  g2.addArc(0, 1, 0);
  g2.addArc(0, 1, 1);
  g2.addArc(0, 1, 2);
  g2.addArc(1, 2, 0);
  g2.addArc(1, 2, 1);
  g2.addArc(1, 2, 2);
  g2.addArc(2, 3, 0);
  g2.addArc(2, 3, 1);
  g2.addArc(2, 3, 2);

  auto composed = compose(g1, g2);
  draw(g1, "simple_compose_g1.dot", symbols);
  draw(g2, "simple_compose_g2.dot", symbols);
  draw(composed, "simple_compose.dot", symbols);
}

// Forwarding WFSAs
void forwardingAcceptors() {
  // The forward algorithm computes the sum of the scores for
  // all accepting paths in a graph. The graph must not have
  // cycles.
  Graph graph;
  graph.addNode(true);
  graph.addNode(true);
  graph.addNode();
  graph.addNode(false, true);

  graph.addArc(0, 1, 0, 0, 1.1);
  graph.addArc(0, 2, 1, 1, 3.2);
  graph.addArc(1, 2, 2, 2, 1.4);
  graph.addArc(2, 3, 0, 0, 2.1);

  // The accepting paths are:
  // 0 2 0 (nodes 0 -> 1 -> 2 -> 3 and score = 1.1 + 1.4 + 2.1)
  // 1 0 (nodes 0 -> 2 0 -> 3 and score = 3.2 + 2.1)
  // 2 0 (nodes 1 -> 2 -> 3 and score = 1.4 + 2.1)
  // The final score is the logadd of the individual path scores.
  auto forwarded = forward(graph);

  // Use Graph::item() to get the score out of a scalar graph:
  float score = forwarded.item();
  std::cout << "The forward score is: " << score << std::endl;

  draw(graph, "simple_forward.dot", symbols);
}

// Differentiable WFSAs
void differentiableAcceptors() {
  // By default a graph will be included in the autograd tape.
  auto in = std::stringstream(
      "0\n"
      "2\n"
      "0 1 0\n"
      "0 1 1\n"
      "1 2 0\n"
      "1 2 1");
  Graph g1 = load(in);
  // To disable gradient computation for and through a graph, set it's
  // calcGrad value to false:
  Graph g2(false);
  g2.addNode(true);
  g2.addNode(false, true);
  g2.addArc(0, 0, 0);
  g2.addArc(0, 1, 1);

  auto a = forward(compose(g1, g2));
  auto b = forward(g1);
  auto loss = subtract(b, a);

  // Differentiate through the computation.
  backward(loss);

  // Access the graph gradient
  Graph grad = g1.grad();
  // The gradient with respect to the input graph arcs are the weights on the
  // arcs of the gradient graph.
  for (auto a = 0; a < grad.numArcs(); ++a) {
    grad.weight(a);
  }
  // The intermediate graphs a and b also have gradients.
  a.grad().weight(0);
  b.grad().weight(0);

  // If gradient computation is disabled, accessing
  // the gradient throws.
  try {
    g2.grad();
  } catch (const std::logic_error& e) {
    std::cout << e.what() << std::endl;
  }


  // Zero the gradients before re-using the graphs in
  // a new computation, otherwise the gradients will
  // simply accumulate.
  g1.zeroGrad();
}

// An example: The Auto Segmentation Criterion
// https://arxiv.org/abs/1609.03193
void autoSegCriterion() {
  // Consider the ASG alignment graph for the sequence
  // [0, 1, 2]
  Graph fal;
  fal.addNode(true);
  fal.addNode();
  fal.addNode();
  fal.addNode(false, true);
  fal.addArc(0, 1, 0);
  fal.addArc(1, 1, 0);
  fal.addArc(1, 2, 1);
  fal.addArc(2, 2, 1);
  fal.addArc(2, 3, 2);
  fal.addArc(3, 3, 2);
  // The fal graph represents all possible alignemnts of the sequence
  // where each token ocurrs one or more times.

  // Now suppose we have an emission graph for an input with 4 frames.
  Graph emissions;
  emissions.addNode(true);
  emissions.addNode();
  emissions.addNode();
  emissions.addNode();
  emissions.addNode(false, true);

  // Loop over time-steps
  for (int t = 0; t < 4; t++) {
    // Loop over alphabet
    for (int i = 0; i < 3; i++) {
      emissions.addArc(t, t + 1, i);
    }
  }

  // To limit the alignments to length 4, we can compose the
  // alignments graph with the emissions graph.
  auto composed = compose(fal, emissions);

  draw(fal, "asg_alignments.dot", symbols);
  draw(emissions, "asg_emissions.dot", symbols);
  draw(composed, "asg_composed.dot", symbols);

  // Compute the asg loss which is the negative log likelihood:
  //                asg = -(fal - fcc)
  // where fal (forward(composed)) is the constrained score and
  // fcc (forward(emissions) is the unconstrained score (i.e.
  // the partition function).
  auto loss = subtract(forward(emissions), forward(composed));

  // To get gradients:
  backward(loss);

  // We can also add transitions by making a bigram transition graph:
  Graph transitions;
  transitions.addNode(true);
  transitions.addNode(false, true);
  transitions.addNode(false, true);
  transitions.addNode(false, true);
  for (int i = 1; i <= 3; i++) {
    transitions.addArc(0, i, i - 1); // p(i | <s>)
  }
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      transitions.addArc(i + 1, j + 1, j); // p(j | i)
    }
  }
  draw(transitions, "asg_transitions.dot", symbols);

  // Computing the actual asg loss with transitions is as simple
  // as composing with the transition graph:
  auto num_graph = compose(compose(fal, transitions), emissions);
  auto denom_graph = compose(emissions, transitions);
  loss = subtract(forward(denom_graph), forward(num_graph));

  // The order of composition won't affect the results as it is an
  // associative operation. However, just like multiplying matrices,
  // the order of operations can make a big difference in run time.
  // For example:
  //     compose(compose(fal, transitions), emissions)
  // will be much faster than
  //     compose(fal, compose(transitions, emissions))
}

// An example: The CTC Criterion
// https://www.cs.toronto.edu/~graves/icml_2006.pdf
void ctcCriterion() {
  std::unordered_map<int, std::string> symbols = {{0, "-"}, {1, "a"}, {2, "b"}};

  // Consider the CTC alignment graph for the sequence
  // [1, 2] where the blank index is 0.
  Graph ctc;
  ctc.addNode(true);
  ctc.addNode();
  ctc.addNode();
  ctc.addNode(false, true);
  ctc.addNode(false, true);

  ctc.addArc(0, 0, 0);
  ctc.addArc(0, 1, 1);
  ctc.addArc(1, 1, 1);
  ctc.addArc(1, 2, 0);
  ctc.addArc(1, 3, 2);
  ctc.addArc(2, 2, 0);
  ctc.addArc(2, 3, 2);
  ctc.addArc(3, 3, 2);
  ctc.addArc(3, 4, 0);
  ctc.addArc(4, 4, 0);

  // The ctc graph represents all possible alignemnts of the sequence
  // where each token ocurrs one or more times with zero or more blank
  // tokens in between.

  // Now suppose we have an emission graph for an input with 4 frames.
  Graph emissions;
  emissions.addNode(true);
  emissions.addNode();
  emissions.addNode();
  emissions.addNode();
  emissions.addNode(false, true);

  // Loop over time-steps
  for (int t = 0; t < 4; t++) {
    // Loop over alphabet (including blank)
    for (int i = 0; i < 3; i++) {
      emissions.addArc(t, t + 1, i);
    }
  }

  // To limit the ctc graph to alignments of length 4, we can compose it
  // with the emissions graph.
  auto composed = compose(ctc, emissions);

  draw(ctc, "ctc_alignments.dot", symbols);
  draw(emissions, "ctc_emissions.dot", symbols);
  draw(composed, "ctc_composed.dot", symbols);

  // Compute the ctc loss
  auto loss = subtract(forward(emissions), forward(composed));
  // In practice, without transitions, we can
  // normalize per frame scores and only compute
  //     loss = negate(forward(composed));

  // We can also add transitions to CTC just like in ASG!
  Graph transitions;
  transitions.addNode(true);
  transitions.addNode(false, true);
  transitions.addNode(false, true);
  transitions.addNode(false, true);
  for (int i = 1; i <= 3; i++) {
    transitions.addArc(0, i, i - 1); // p(i | <s>)
  }
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      transitions.addArc(i + 1, j + 1, j); // p(j | i)
    }
  }

  // Computing the ctc loss is identical to computing the asg loss,
  // the only difference is the alignment graph (ctc instead of asg).
  auto num_graph = compose(compose(ctc, transitions), emissions);
  auto denom_graph = compose(emissions, transitions);
  loss = subtract(forward(denom_graph), forward(num_graph));
}

void simpleTransducers() {
  Graph graph;
  graph.addNode(true);
  graph.addNode();
  graph.addNode(false, true);

  // By default a graph is an acceptor
  assert(graph.acceptor());

  // Adding an arc with just an input label, the output label defaults to have
  // the same value as the input label
  graph.addArc(0, 1, 0);

  // Add an arc from node 0 to 2 with the same input and output label of 1
  graph.addArc(0, 1, 1, 1);

  // The graph is still an acceptor
  assert(graph.acceptor());

  // However, adding an arc with a different input and output label
  graph.addArc(1, 2, 1, 2);

  // The graph is no longer an acceptor
  assert(!graph.acceptor());

  // Specify the input and output symbols
  draw(graph, "simple_fst.dot", isymbols, osymbols);
}

// Composing WFSTs
void composingTransducers() {
  // The composition of two trandsucers is the graph which represents the set
  // of all paths such that the output of labelling of a path in the first
  // graph matches the input labelling of a path in the second graph. The
  // labelling of the path is the input labelling of the path in the first
  // graph and the output labelling of the path in the second graph. The score
  // of the path in the composed graph is the sum of the scores for the paths
  // in the two input graphs.
  Graph g1;
  g1.addNode(true);
  g1.addNode(false, true);
  g1.addArc(0, 0, 0, 0);
  g1.addArc(0, 1, 1, 1);
  g1.addArc(1, 1, 2, 2);

  Graph g2;
  g2.addNode(true);
  g2.addNode();
  g2.addNode();
  g2.addNode(false, true);
  g2.addArc(0, 1, 0, 0);
  g2.addArc(0, 1, 0, 1);
  g2.addArc(0, 1, 1, 2);
  g2.addArc(1, 2, 0, 0);
  g2.addArc(1, 2, 1, 1);
  g2.addArc(1, 2, 2, 2);
  g2.addArc(2, 3, 1, 0);
  g2.addArc(2, 3, 2, 1);
  g2.addArc(2, 3, 2, 2);

  // The output alphabet of the first graph is assumed to be the same as as the
  // input alphabet of the second graph. Note also that composing/intersecting
  // two acceptors commutes, but composing a transducer with another transducer
  // or an acceptor does not.
  auto composed = compose(g1, g2);
  draw(g1, "transducer_compose_g1.dot", isymbols, osymbols);
  draw(g2, "transducer_compose_g2.dot", osymbols, isymbols);
  draw(composed, "transducer_compose.dot", isymbols, isymbols);
}

// WFSTs with epsilons
void epsilonTransitions() {
  // Transducers or acceptors can have epsilon transitions.
  Graph g1;
  g1.addNode(true);
  g1.addNode();
  g1.addNode(false, true);

  // Use Graph::epsilon to denote an epsilon label (the integer value is -1,
  // though you should avoid using that directly to make your code more future
  // proof).
  g1.addArc(0, 1, 1, Graph::epsilon, 1.1);
  g1.addArc(1, 2, 0, 0, 2);

  // We can forward graphs with epsilons (as long as they don't have any
  // cycles).
  forward(g1);

  g1.addArc(0, 0, 0, Graph::epsilon, 0.5);

  // Drawing will use a special "ε" token to represent
  // `Graph::epsilon` when symbols are specified.
  draw(g1, "epsilon_graph1.dot", isymbols, osymbols);

  Graph g2;
  g2.addNode(true);
  g2.addNode(false, true);
  g2.addArc(0, 1, 0, 0, 1.3);
  g2.addArc(1, 1, Graph::epsilon, 2, 2.5);
  draw(g2, "epsilon_graph2.dot", osymbols, isymbols);

  // We can compose graphs with epsilons
  auto composed = compose(g1, g2);
  draw(composed, "epsilon_composed.dot", isymbols, isymbols);

  // For a detailed discussion on composition with epsilon transitions see
  // "Weighted Automata Algorithms", Mehryar Mohri,
  // https://cs.nyu.edu/~mohri/pub/hwa.pdf, Section 5.1
}

int main() {
  simpleAcceptors();
  interestingAcceptors();
  simpleOps();
  composingAcceptors();
  forwardingAcceptors();
  differentiableAcceptors();
  autoSegCriterion();
  ctcCriterion();
  simpleTransducers();
  composingTransducers();
  epsilonTransitions();
}
