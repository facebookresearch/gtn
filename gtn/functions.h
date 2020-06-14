#pragma once

#include "graph.h"

namespace gtn {

/* Scalar only functions */
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

/* Performs a deep clone of a graph with an option to project to either the
 * input or output labels. The operation is NOT recorded in the autograd tape.
 * For a version of this function that does record on autograd tape, see
 * `clone`.
 */
Graph copy(Graph other, Projection projection = Projection::NONE);

void copy(Graph other, Projection projection, Graph& out);

/* Removes the output labels from the graph and records the operation in the
 * autograd tape. This function makes a copy of the input graph. */
Graph projectInput(Graph other);

/* Removes the input labels from the graph and records the operation in the
 * autograd tape. This function makes a copy of the input graph. */
Graph projectOutput(Graph other);

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
