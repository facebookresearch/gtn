#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import gtn

# Recognizes "aba*"
g1 = gtn.Graph(False)
g1.add_node(True)
g1.add_node()
g1.add_node(False, True)
g1.add_arc(0, 1, 0)
g1.add_arc(1, 2, 1)
g1.add_arc(2, 2, 0)

# Recognizes "ba"
g2 = gtn.Graph(False)
g2.add_node(True)
g2.add_node()
g2.add_node(False, True)
g2.add_arc(0, 1, 1)
g2.add_arc(1, 2, 0)

# Recognizes "ac"
g3 = gtn.Graph(False)
g3.add_node(True)
g3.add_node()
g3.add_node(False, True)
g3.add_arc(0, 1, 0)
g3.add_arc(1, 2, 2)

symbols = {0: "a", 1: "b", 2: "c"}

gtn.draw(g1, "/tmp/union_g1.pdf", symbols, symbols)
gtn.draw(g2, "/tmp/union_g2.pdf", symbols, symbols)
gtn.draw(g3, "/tmp/union_g3.pdf", symbols, symbols)

graph = gtn.union([g1, g2, g3])

gtn.draw(graph, "/tmp/union_graph.pdf", symbols, symbols)
