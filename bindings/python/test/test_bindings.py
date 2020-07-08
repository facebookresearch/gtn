#!/usr/bin/env python3

import unittest
from test_utils import GTNModuleTestCase

try:
    from gtn import *
except ImportError:
    print("Could not import gtn package - will skip tests")


class GraphTestCase(GTNModuleTestCase):
    def setUp(self):
        g = Graph(False)
        g.add_node(True)
        g.add_node()
        g.add_node()
        g.add_node()
        g.add_node(False, True)

        g.add_arc(0, 1, 0)
        g.add_arc(0, 2, 1)
        g.add_arc(1, 2, 0)
        g.add_arc(1, 1, 1, 1, 2.1)
        g.add_arc(2, 3, 2)
        
        self.g = g

    def test_graph_basic(self):
        self.assertEqual(self.g.num_nodes(), 5)
        self.assertEqual(self.g.num_start(), 1)
        self.assertEqual(self.g.num_accept(), 1)

        self.assertEqual(self.g.num_arcs(), 5)

    def test_graph_weights(self):
        weights = self.g.weights()
        expected = [0, 0, 0, 2.1, 0]
        self.assertEqual(len(weights), len(expected))
        for i in range(0, len(weights)):
            self.assertAlmostEqual(weights[i], expected[i], places=4)
