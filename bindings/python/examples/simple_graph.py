#!/usr/bin/env python3

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

gtn.draw(g1, "/tmp/sum_g1.dot", symbols, symbols)
gtn.draw(g2, "/tmp/sum_g2.dot", symbols, symbols)
gtn.draw(g3, "/tmp/sum_g3.dot", symbols, symbols)

graph = gtn.sum([g1, g2, g3])

gtn.draw(graph, "/tmp/sum_graph.dot", symbols, symbols)
