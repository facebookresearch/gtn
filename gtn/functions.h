/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "graph.h"

namespace gtn {

/** \addtogroup functions
 *  @{
 */

/** Negate a scalar graph. */
Graph negate(const Graph& g);
/** Add two scalar graphs. */
Graph add(const Graph& g1, const Graph& g2);
/** Subtract one scalar graph from another. */
Graph subtract(const Graph& g1, const Graph& g2);

/**
 * Projection type used with `gtn::clone`.
 */
enum class Projection {
  /** Keep both input and output labels. */
  NONE = 0,
  /** Keep only the input lables. */
  INPUT = 1,
  /** Keep only the output lables. */
  OUTPUT = 2,
};

/**
 * Performs a deep clone of a graph with an option to project to either the
 * input or output labels. The operation is recorded in the autograd tape. For a
 * version which is not recorded on the autograd tape, see `Graph::deepCopy`.
 */
Graph clone(const Graph& g, Projection projection = Projection::NONE);

/**
 * Removes the output labels from the graph and records the operation in the
 * autograd tape. This function makes a copy of the input graph.
 */
Graph projectInput(const Graph& g);

/**
 * Removes the input labels from the graph and records the operation in the
 * autograd tape. This function makes a copy of the input graph.
 */
Graph projectOutput(const Graph& g);

/**
 * Create the concatenation of two graphs. This operation is recorded in the
 * autograd tape.
 *
 * Equivalent to `concat({g1, g2})`, see
 * `gtn::concat(const std::vector<Graph>&)`.
 */
Graph concat(const Graph& g1, const Graph& g2);

/**
 * Concatenate a vector of graphs. This operation is recorded in the autograd
 * tape.
 *
 * If `x_i` is a sequence accepted (or `x_i:y_i` is transduced) by `graphs[i]`
 * then the concatenated graph accepts the sequence `x_1x_2...x_n` if `graphs`
 * contains `n` graphs. The score of the path `x_1...x_n` is the sum of the
 * scores of the individual `x_i` in `graphs[i]`. The concatenated graph is
 * constructuted by connecting every accepting state of `graphs[i-1]` to every
 * starting state of `graphs[i]` with an epsilon transition. The starting state
 * of the concatenated graphs are starting states of `graphs[0]` and the
 * accepting states are accepting states of `graphs.back()`.
 *
 * Note the concatenation of 0 graphs `gtn::concat({})` is the graph which
 * accepts the empty string (epsilon). The concatentation of a single graph is
 * equivalent to a clone.
 */
Graph concat(const std::vector<Graph>& graphs);

/** Create the (Kleene) closure of the graph. */
Graph closure(const Graph& g);

/** Create the union of a vector of graphs. */
Graph union_(const std::vector<Graph>& graphs);

/**
 * Create the equivalent graph without epsilon transitions. If label is
 * specified then instead of removing epsilon transitions, arcs with the
 * matching label are removed. The removed arc labels are treated as if they
 * were epsilon transitions. Note this is different than simply pruning the
 * arc.
 */
Graph remove(const Graph& g, int label = epsilon);

/**
 * Create the equivalent graph without `ilabel:olabel` transitions. The removed
 * arc labels are treated as if they were epsilon transitions. Note this is
 * different than simply pruning the arc.
 */
Graph remove(const Graph& g, int ilabel, int olabel);

/**
 * Compose two transducers. This operation is recorded in the autograd tape.
 * If x:y is transduced by `g1` and `y:z` is transduced by `g2` then the
 * composition will transduce `x:z`. The arc scores are added in the composed
 * graph.
 */
Graph compose(const Graph& g1, const Graph& g2);

/**
 * Intersect two acceptors. This operation is recorded in the autograd tape.
 * This function only works on acceptors, calling it on a `graph` where
 * `graph.ilabel(a) != graph.olabel(a)` for some `a` is undefined and may yield
 * incorrect results. The intersected graph accepts any path `x` which is
 * accepted by both `g1` and `g2`. The arc scores are added in the
 * intersected graph.
 *
 * The result of `gtn::compose` will yield an equivalent result, however; this
 * function should be preferred since the implementation may be faster.
 */
Graph intersect(const Graph& g1, const Graph& g2);

/**
 * Compute the forward score of a graph. Returns the score in a scalar graph
 * which can be accessed with `Graph::item()`. This is equivalent to the
 * shortest distance from the start nodes to the accept nodes in the log
 * semiring.
 * NB: This assumes the input graph is acyclic.
 */
Graph forwardScore(const Graph& g);

/**
 * Compute the Viterbi score of a graph. Returns the score in a scalar graph
 * which can be accessed with `Graph::item()`. This is equivalent to the
 * shortest distance from the start nodes to the accepting nodes in the
 * tropical semiring.
 * NB: This assumes the input graph is acyclic.
 */
Graph viterbiScore(const Graph& g);

/**
 * Compue the Viterbi shortest path of a graph and return it in a single
 * chain graph with the labels and weights of the shortest path. This is
 * equivalent to the shortest path from the start nodes to the accepting nodes
 * in the tropical semiring.
 * NB: This assumes the input graph is acyclic.
 */
Graph viterbiPath(const Graph& g);

/** @} */
} // namespace gtn
