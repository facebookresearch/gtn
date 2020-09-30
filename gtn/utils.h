/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "gtn/graph.h"

namespace gtn {

/** \addtogroup comparisons
 *  @{
 */

/** Checks if two graphs are exactly equal (not isomorphic). */
bool equal(const Graph& g1, const Graph& g2);

/**
 * Checks if two graphs are isomorphic. Note this function will be very very
 * slow for large graphs.
 */
bool isomorphic(const Graph& g1, const Graph& g2);

/** @}*/

/** \addtogroup input_output
 *  @{
 */

/** Save a graph to a file in binary format. */
void save(const std::string& fileName, const Graph& g);

/** Save a graph to an output stream in binary format. */
void save(std::ostream& out, const Graph& g);

/** Load a graph from a file in binary format. */
Graph load(const std::string& fileName);

/** Load a graph from an input stream in binary format. */
Graph load(std::istream& in);

/** Load a graph from an input stream in binary format. */
Graph load(std::istream&& in);

/**
 * Save a graph to a file in txt format.
 *
 * The first line will be a space separated list of the start nodes
 * (`Graph::start`) and the second line will be a space separated list of
 * accepting nodes (`Graph::accept`).The following lines contain the arcs in
 * the format:
 * ```
 * srcNode dstNode ilabel olabel weight
 * ```
 * 
 */
void saveTxt(const std::string& fileName, const Graph& g);

/** Save a graph to an output stream in txt format. */
void saveTxt(std::ostream& out, const Graph& g);

/**
 * Load a graph from a file in txt format.
 *
 * The expected format is the first two lines contain a list of space separated
 * start and accept nodes respectively. The following lines should contain the
 * arcs in the format:
 *   ```
 *   srcNode dstNode ilabel [olabel=ilabel] [weight=0.0]
 *   ```
 * where `[x=y]` indicate optional values for `x` with a default value of `y`.
 *
 * For example:
 * ```
 *   0
 *   1
 *   0 1 1
 * ```
 * is a two node graph with an arc from start node 0 to accept node 1 with
 * input and output label of 1,
 * ```
 *   0
 *   1
 *   0 1 1 2
 * ```
 * is a two node graph with an arc from node 0 to node 1 with input label 1 and
 * output label 2, and
 * ```
 *   0
 *   1
 *   0 1 1 2 3.0
 * ```
 * is a two node graph with an arc from node 0 to node 1 with input label 1,
 * output label 2, and a weight of 3.0.
 *
 * The returned Graph will contain "maxNodeIdx" nodes where "maxNodeIdx" is the
 * largest index of a node in the file regardless of if there are arcs for every
 * node.
 */
Graph loadTxt(const std::string& fileName);

/** Load a graph from an input stream in txt format. */
Graph loadTxt(std::istream& in);

/** Load a graph from an input stream in txt format. */
Graph loadTxt(std::istream&& in);

/** overload `<<` operator for working with `std::cout` on `Graph` */
std::ostream& operator<<(std::ostream& out, const Graph& g);

/**
 * User defined map of integer to arc label strings for use with e.g.
 * `gtn::draw`.
 */
using SymbolMap = std::unordered_map<int, std::string>;

/**
 * Write a graph in the [Graphviz](https://graphviz.org/) DOT format to a file.
 * Arc labels are of the format `ilabel/olabel:weight`. If the output symbols
 * are not specified then the `olabel` is omitted and arc labels are of the
 * format `ilabel:weight`. If the input symbols are not specified then integer
 * ids are used as the label.
 *
 * Compile to pdf with:
 * ```
 * dot -Tpdf graph.dot -o graph.pdf
 * ```
 * @param g The graph to draw
 * @param filename The name of the file to write to (e.g. graph.dot).
 * @param isymbols A map of integer ids to strings used for arc input labels
 * @param osymbols A map of integer ids to strings used for arc output labels
 */
void draw(
    const Graph& g,
    const std::string& filename,
    const SymbolMap& isymbols = SymbolMap(),
    const SymbolMap& osymbols = SymbolMap());

/**
 * Write a graph in graphviz dot format to an output stream. See `gtn::draw`.
 */
void draw(
    const Graph& g,
    std::ostream& out,
    const SymbolMap& isymbols = SymbolMap(),
    const SymbolMap& osymbols = SymbolMap());

/** @}*/

} // namespace gtn
