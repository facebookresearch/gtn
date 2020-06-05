#!/usr/bin/env python3

from gtn.graph import Graph 
from gtn.utils import draw
from gtn.functions import sum

# Recognizes "aba*"
g1= Graph(False)
g1.add_node(True)
g1.add_node()
g1.add_node(False, True)
g1.add_arc(0, 1, 0)
g1.add_arc(1, 2, 1)
g1.add_arc(2, 2, 0)

# Recognizes "ba"
g2 = Graph(False)
g2.add_node(True)
g2.add_node()
g2.add_node(False, True)
g2.add_arc(0, 1, 1)
g2.add_arc(1, 2, 0)

# Recognizes "ac"
g3 = Graph(False)
g3.add_node(True)
g3.add_node()
g3.add_node(False, True)
g3.add_arc(0, 1, 0)
g3.add_arc(1, 2, 2)

symbols = {0: "a", 1: "b", 2: "c"}

draw(g1, "/tmp/sum_g1.dot", symbols, symbols)
draw(g2, "/tmp/sum_g2.dot", symbols, symbols)
draw(g3, "/tmp/sum_g3.dot", symbols, symbols)

graph = sum([g1, g2, g3])

draw(graph, "/tmp/sum_graph.dot", symbols, symbols)