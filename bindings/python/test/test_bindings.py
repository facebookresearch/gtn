#!/usr/bin/env python3

import unittest
from test_utils import GTNModuleTestCase

try:
    from gtn import *
except ImportError:
    print("Could not import gtn package - will skip tests")


class GraphTestCase(GTNModuleTestCase):
    def test_graph(self):
        g = Graph(False)
        g.add_node(True)
        g.add_node()
        g.add_node()
        g.add_node()
        g.add_node(False, True)
        self.assertEqual(g.num_nodes(), 5)
        self.assertEqual(g.num_start(), 1)
        self.assertEqual(g.num_accept(), 1)

        g.add_arc(0, 1, 0)
        g.add_arc(0, 2, 1)
        g.add_arc(1, 2, 0)
        g.add_arc(1, 1, 1, 1, 2.1)
        g.add_arc(2, 3, 2)

        self.assertEqual(g.num_arcs(), 5)
