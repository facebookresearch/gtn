"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import gtn

from time_utils import time_func


def time_compose():
    N1 = 100
    N2 = 50
    A1 = 20
    A2 = 500
    graphs1 = [gtn.linear_graph(N1, A1) for _ in range(B)]
    graphs2 = [gtn.linear_graph(N2, A2) for _ in range(B)]
    for g in graphs2:
        for i in range(N2):
            for j in range(A2):
                g.add_arc(i, i, j)

    def fwd():
        gtn.compose(graphs1, graphs2)

    time_func(fwd, 20, "parallel compose Fwd")

    out = gtn.compose(graphs1, graphs2)

    def bwd():
        gtn.backward(out, [True])

    time_func(bwd, 20, "parallel compose bwd")


def time_forward_score():
    graphs = [gtn.linear_graph(1000, 100) for _ in range(B)]

    def fwd():
        gtn.forward_score(graphs)

    time_func(fwd, 100, "parallel forward_score Fwd")

    out = gtn.forward_score(graphs)

    def bwd():
        gtn.backward(out, [True])

    time_func(bwd, 100, "parallel forward_score bwd")


def time_indexed_func():
    N1 = 100
    N2 = 50
    A1 = 20
    A2 = 500
    graphs1 = [gtn.linear_graph(N1, A1) for _ in range(B)]
    graphs2 = [gtn.linear_graph(N2, A2) for _ in range(B)]
    for g in graphs2:
        for i in range(N2):
            for j in range(A2):
                g.add_arc(i, i, j)

    out = [None] * B

    def process(b):
        out[b] = gtn.forward_score(gtn.compose(graphs1[b], graphs2[b]))

    def indexed_func():
        gtn.parallel_for(process, range(B))

    time_func(indexed_func, 100, "parallel indexed python func")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        B = int(sys.argv[1])
    else:
        B = 1
    time_compose()
    time_forward_score()
    time_indexed_func()
