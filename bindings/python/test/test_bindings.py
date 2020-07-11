#!/usr/bin/env python3

from test_utils import GTNModuleTestCase

try:
    import gtn
except ImportError:
    print("Could not import gtn package - will skip tests")


class GraphTestCase(GTNModuleTestCase):
    def setUp(self):
        g = gtn.Graph(False)
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

    def test_graph_weights_get(self):
        weights = self.g.weights()
        expected = [0, 0, 0, 2.1, 0]
        self.assertEqual(len(weights), len(expected))
        for i in range(0, len(weights)):
            self.assertAlmostEqual(weights[i], expected[i], places=4)

    def test_graph_weights_set(self):
        weights_original = self.g.weights()
        weights_new_expected = [1.1, -3.4, 0, 0.5, 0]
        self.g.set_weights(weights_new_expected)
        weights_new = self.g.weights()
        self.assertEqual(len(weights_new), len(weights_new_expected))
        for i in range(0, len(weights_new)):
            self.assertAlmostEqual(weights_new[i], weights_new_expected[i], places=4)
        self.g.set_weights(weights_original)


class FunctionsTestCase(GTNModuleTestCase):

    def test_scalar_ops(self):
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0, 0, 1.0)

        # Test negate:
        res = gtn.negate(g1)
        self.assertEqual(res.item(), -1.0)
        gtn.backward(res)
        self.assertEqual(g1.grad().item(), -1.0)
        g1.zero_grad()

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 0, 3.0)

        # Test add:
        res = gtn.add(g1, g2)
        self.assertEqual(res.item(), 4.0)
        gtn.backward(res)
        self.assertEqual(g1.grad().item(), 1.0)
        self.assertEqual(g2.grad().item(), 1.0)
        g1.zero_grad()
        g2.zero_grad()

        # Test subtract:
        res = gtn.subtract(g1, g2)
        self.assertEqual(res.item(), -2.0)
        gtn.backward(res)
        self.assertEqual(g1.grad().item(), 1.0)
        self.assertEqual(g2.grad().item(), -1.0)
