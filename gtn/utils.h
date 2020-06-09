#pragma once

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "graph.h"

namespace gtn {

/* Checks if two graphs are exactly equal (not isomorphic). */
bool equals(Graph a, Graph b);

/* Checks if two graphs are isomorphic. Note this function will be very very
 * slow for large graphs. */
bool isomorphic(Graph a, Graph b);

/* Load a graph from a file.
 *
 * The expected format is the first two lines contain a list of space separated
 * start and accept nodes respectively. The following lines contain the arcs in
 * the format:
 *   <src> <dest> <ilabel> [olabel=ilabel] [weight=0.0]
 * where [x=y] indicate optional values for x with a default value of y.
 *
 * For example:
 *   0
 *   1
 *   0 1 1
 * is a two node graph with an arc from start node 0 to accept node 1 with
 * ilabel 1,
 *   0
 *   1
 *   0 1 1 2
 * is a two node graph with an arc from node 0 to node 1 with ilabel 1 and
 * olabel 2, and
 *   0
 *   1
 *   0 1 1 2 3.0
 * is a two node graph with an arc from node 0 to node 1 with ilabel 1, olabel
 * 2 and a weight of 3.0.
 */
Graph load(const std::string& fileName);

/* Load a graph from an input stream. */
Graph load(std::istream& in);
Graph load(std::istream&& in);

/* Prints a graph to stdout as a list of arcs:
 * <up node> <down node> <arc label> <arc weight>
 * the first line will be a space separated list
 * of the start nodes and the second line will be a
 * space separated list of accepting nodes.
 */
void print(Graph graph);

/* Prints a graph to an output stream. */
void print(Graph graph, std::ostream& out);

/* Draw a graph in graphviz dot format.
 * Arc labels are of the format "ilabel/olabel:weight". If the graph is an
 * acceptor and the output symbols are not specified then the olabel is omitted
 * "ilabel:weight"
 * */
using SymbolMap = std::unordered_map<int, std::string>;
void draw(
    Graph graph,
    std::ostream& out,
    const SymbolMap& isymbols = SymbolMap(),
    const SymbolMap& osymbols = SymbolMap());
void draw(
    Graph graph,
    const std::string& filename,
    const SymbolMap& isymbols = SymbolMap(),
    const SymbolMap& osymbols = SymbolMap());

/*
 * Compute fwd and bwd for a given criterion graph
 */
void sequenceCriterion(
    float* input,
    Graph criterion,
    int numTimeSteps,
    int numFeatures,
    std::vector<int> labels,
    std::vector<float> lossScales,
    float* loss,
    float* inputGrad
);

} // namespace gtn
