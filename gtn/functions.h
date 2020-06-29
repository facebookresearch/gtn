#pragma once

#include "graph.h"

namespace gtn {

/* Scalar only functions */
Graph negate(Graph other);
Graph add(Graph lhs, Graph rhs);
Graph subtract(Graph lhs, Graph rhs);

/* Graph-level operations */

/**
 * Projeciton type
 */
enum class Projection {
  NONE = 0,
  INPUT = 1,
  OUTPUT = 2,
};

/* Performs a deep clone of a graph with an option to project to either the
 * input or output labels. The operation is recorded in the autograd tape. For a
 * version which is not recorded on the autograd tape, see `copy`. */
Graph clone(Graph other, Projection projection = Projection::NONE);

/* Removes the output labels from the graph and records the operation in the
 * autograd tape. This function makes a copy of the input graph. */
Graph projectInput(Graph other);

/* Removes the input labels from the graph and records the operation in the
 * autograd tape. This function makes a copy of the input graph. */
Graph projectOutput(Graph other);

/* Create the concatenation of two graphs. This operation is recorded in the
 * autograd tape.
 *
 * Equivalent to `concat({lhs, rhs})`, see:
 *   `Graph concat(std::vector<Graph> * graphs)`.
 **/
Graph concat(Graph lhs, Graph rhs);

/* Create the concatenation of a vector of graphs. This operation is recorded
 * in the autograd tape.
 *
 * If `x_i` is a sequence accepted (or `x_i:y_i` is transduced) by `graphs[i]`
 * then the concatenated graph accepts the sequence `x_1x_2...x_n` if `graphs`
 * contains `n` graphs. The score of the path `x_1...x_n` is the sum of the
 * scores of the individual `x_i` in `graphs[i]`. The concatenated graph is
 * constructuted by connected every accepting state of `graphs[i-1]` to every
 * starting state of `graphs[i]` with an epsilon transition. The starting state
 * of the concatenated graphs are starting states of `graphs[0]` and the
 * accepting states are accepting states of `graphs.back()`.
 *
 * Note the concatenation of 0 graphs `concat({})` is the graph which accepts
 * the empty string (epsilon). The concatentation of a single graph is
 * equivalent to a clone.
 */
Graph concat(std::vector<Graph> graphs);

/* Create the (Kleene) closure of the graph. */
Graph closure(Graph graph);

/* Create the sum (union) of a vector of graphs. */
Graph sum(std::vector<Graph> graphs);

// Create the equivalent graph without epsilon transitions. If labels are
// specified then instead of removing epsilon transitions, arcs with the
// matching label are removed. The removed arc labels are treated as if they
// were epsilon transitions. Note this is different than pruning the arc.
Graph remove(Graph other, int label = Graph::epsilon);
Graph remove(Graph other, int ilabel, int olabel);

// Compose two graphs.
Graph compose(Graph first, Graph second);

// Compute the forward score of a graph. This may only be
// computed on acyclic graphs without self-loops
Graph forward(Graph graph);
} // namespace gtn
