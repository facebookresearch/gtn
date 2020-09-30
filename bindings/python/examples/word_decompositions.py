#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import gtn


def lexicon_graph(word_pieces, letters_to_idx):
    """
    Constructs a graph which transudces letters to word pieces.
    """
    lex = []
    for i, wp in enumerate(word_pieces):
        graph = gtn.Graph()
        graph.add_node(True)
        for e, l in enumerate(wp):
            if e == len(wp) - 1:
                graph.add_node(False, True)
                graph.add_arc(e, e + 1, letters_to_idx[l], i)
            else:
                graph.add_node()
                graph.add_arc(e, e + 1, letters_to_idx[l], gtn.epsilon)
        lex.append(graph)
    return gtn.closure(gtn.union(lex))


def token_graph(token_list):
    """
    Constructs a graph with all the individual
    token transition models.
    """
    tokens = []
    for i, wp in enumerate(token_list):
        # We can consume one or more consecutive
        # word pieces for each emission:
        # E.g. [ab, ab, ab] transduces to [ab]
        graph = gtn.Graph()
        graph.add_node(True)
        graph.add_node(False, True)
        graph.add_arc(0, 1, i, i)
        graph.add_arc(1, 1, i, gtn.epsilon)
        tokens.append(graph)
    return gtn.closure(gtn.union(tokens))


if __name__ == "__main__":
    # letter to index map:
    let_to_idx= {"a" : 0, "b" : 1, "c" : 2}
    idx_to_let = {v : k for k, v in let_to_idx.items()}

    # set of allowed word pieces:
    word_pieces = ["a", "b", "c", "ab", "bc", "ac", "abc"]
    idx_to_wp = dict(enumerate(word_pieces))

    lex = lexicon_graph(word_pieces, let_to_idx)
    gtn.draw(lex, "lexicon.pdf", idx_to_let, idx_to_wp)

    # Build the token graph:
    tokens = token_graph(word_pieces)
    gtn.draw(tokens, "tokens.pdf", idx_to_wp, idx_to_wp)

    # Recognizes "abc":
    abc = gtn.Graph(False)
    abc.add_node(True)
    abc.add_node()
    abc.add_node()
    abc.add_node(False, True)
    abc.add_arc(0, 1, let_to_idx["a"])
    abc.add_arc(1, 2, let_to_idx["b"])
    abc.add_arc(2, 3, let_to_idx["c"])
    gtn.draw(abc, "abc.pdf", idx_to_let)

    # Compute the decomposition graph for "abc":
    abc_decomps = gtn.remove(gtn.project_output(gtn.compose(abc, lex)))
    gtn.draw(abc_decomps, "abc_decomps.pdf", idx_to_wp, idx_to_wp)

    # Compute the alignment graph for "abc":
    abc_alignments = gtn.project_input(
        gtn.remove(gtn.compose(tokens, abc_decomps)))
    gtn.draw(abc_alignments, "abc_alignments.pdf", idx_to_wp)

    # From here we can use the alignment graph with an emissions graph and
    # transitions graphs to compute the sequence level criterion:
    emissions = gtn.linear_graph(10, len(word_pieces), True)
    loss = gtn.subtract(
        gtn.forward_score(emissions),
        gtn.forward_score(gtn.intersect(emissions, abc_alignments)))
    print(f"Loss is {loss.item():.2f}")
