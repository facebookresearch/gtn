import math
import unittest
import gtn
from test_helpers import create_graph_from_text


class FunctionsTestCase(unittest.TestCase):
    def test_scalar_ops(self):
        g1 = gtn.scalar_graph(3.0)

        result = gtn.negate(g1)
        self.assertEqual(result.item(), -3.0)

        g2 = gtn.scalar_graph(4.0)

        result = gtn.add(g1, g2)
        self.assertEqual(result.item(), 7.0)

        result = gtn.subtract(g2, g1)
        self.assertEqual(result.item(), 1.0)

    def test_project_clone(self):

        g_str = [
            "0 1",
            "3 4",
            "0 1 0 0 2",
            "0 2 1 1 1",
            "1 2 0 0 2",
            "2 3 0 0 1",
            "2 3 1 1 1",
            "1 4 0 0 2",
            "2 4 1 1 3",
            "3 4 0 0 2",
        ]
        graph = create_graph_from_text(g_str)

        # Test clone
        cloned = gtn.clone(graph)
        self.assertTrue(gtn.equal(graph, cloned))

        # Test projecting input
        g_str = [
            "0 1",
            "3 4",
            "0 1 0 0 2",
            "0 2 1 1 1",
            "1 2 0 0 2",
            "2 3 0 0 1",
            "2 3 1 1 1",
            "1 4 0 0 2",
            "2 4 1 1 3",
            "3 4 0 0 2",
        ]
        inputExpected = create_graph_from_text(g_str)
        self.assertTrue(gtn.equal(gtn.project_input(graph), inputExpected))

        # Test projecting output
        g_str = [
            "0 1",
            "3 4",
            "0 1 0 0 2",
            "0 2 1 1 1",
            "1 2 0 0 2",
            "2 3 0 0 1",
            "2 3 1 1 1",
            "1 4 0 0 2",
            "2 4 1 1 3",
            "3 4 0 0 2",
        ]
        outputExpected = create_graph_from_text(g_str)
        self.assertTrue(gtn.equal(gtn.project_output(graph), outputExpected))

    def test_composition(self):
        # Compos,ing with an empty graph gives an empty graph
        g1 = gtn.Graph()
        g2 = gtn.Graph()
        self.assertTrue(gtn.equal(gtn.compose(g1, g2), gtn.Graph()))

        g1.add_node(True)
        g1.add_node()
        g1.add_arc(0, 1, 0)

        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0)
        g2.add_arc(0, 1, 0)

        self.assertTrue(gtn.equal(gtn.compose(g1, g2), gtn.Graph()))
        self.assertTrue(gtn.equal(gtn.compose(g2, g1), gtn.Graph()))
        self.assertTrue(gtn.equal(gtn.intersect(g2, g1), gtn.Graph()))

        # Check singly sorted version
        g1.arc_sort(True)
        self.assertTrue(gtn.equal(gtn.compose(g1, g2), gtn.Graph()))

        # Check doubly sorted version
        g2.arc_sort()
        self.assertTrue(gtn.equal(gtn.compose(g1, g2), gtn.Graph()))

        # Self-loop in the composed graph
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 0, 0)
        g1.add_arc(0, 1, 1)
        g1.add_arc(1, 1, 2)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node()
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0)
        g2.add_arc(1, 1, 0)
        g2.add_arc(1, 2, 1)

        g_str = ["0", "2", "0 1 0", "1 1 0", "1 2 1"]
        expected = create_graph_from_text(g_str)
        self.assertTrue(gtn.isomorphic(gtn.compose(g1, g2), expected))
        self.assertTrue(gtn.isomorphic(gtn.intersect(g1, g2), expected))

        # Check singly sorted version
        g1.arc_sort(True)
        self.assertTrue(gtn.isomorphic(gtn.compose(g1, g2), expected))

        # Check doubly sorted version
        g2.arc_sort()
        self.assertTrue(gtn.isomorphic(gtn.compose(g1, g2), expected))

        # Loop in the composed graph
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0)
        g1.add_arc(1, 1, 1)
        g1.add_arc(1, 0, 0)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 0, 0)
        g2.add_arc(0, 1, 1)
        g2.add_arc(1, 0, 1)

        g_str = ["0", "2", "0 1 0", "1 0 0", "1 2 1", "2 1 1"]
        expected = create_graph_from_text(g_str)
        self.assertTrue(gtn.isomorphic(gtn.compose(g1, g2), expected))
        self.assertTrue(gtn.isomorphic(gtn.intersect(g1, g2), expected))

        # Check singly sorted version
        g1.arc_sort(True)
        self.assertTrue(gtn.isomorphic(gtn.compose(g1, g2), expected))

        # Check doubly sorted version
        g2.arc_sort()
        self.assertTrue(gtn.isomorphic(gtn.compose(g1, g2), expected))

        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node()
        g1.add_node()
        g1.add_node()
        g1.add_node(False, True)
        for i in range(g1.num_nodes() - 1):
            for j in range(3):
                g1.add_arc(i, i + 1, j, j, j)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node()
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 0, 3.5)
        g2.add_arc(1, 1, 0, 0, 2.5)
        g2.add_arc(1, 2, 1, 1, 1.5)
        g2.add_arc(2, 2, 1, 1, 4.5)
        g_str = [
            "0",
            "6",
            "0 1 0 0 3.5",
            "1 2 0 0 2.5",
            "1 4 1 1 2.5",
            "2 3 0 0 2.5",
            "2 5 1 1 2.5",
            "4 5 1 1 5.5",
            "3 6 1 1 2.5",
            "5 6 1 1 5.5",
        ]
        expected = create_graph_from_text(g_str)
        self.assertTrue(gtn.isomorphic(gtn.compose(g1, g2), expected))
        self.assertTrue(gtn.isomorphic(gtn.intersect(g1, g2), expected))

        # Check singly sorted version
        g1.arc_sort(True)
        self.assertTrue(gtn.isomorphic(gtn.compose(g1, g2), expected))

        # Check doubly sorted version
        g2.arc_sort()
        self.assertTrue(gtn.isomorphic(gtn.compose(g1, g2), expected))

    def test_forward(self):

        # Check score of empty graph
        g = gtn.Graph()
        self.assertEqual(gtn.forward_score(g).item(), -math.inf)

        # Throws on self-loops
        g = gtn.Graph()
        g.add_node(True, True)
        g.add_arc(0, 0, 1)
        self.assertRaises(ValueError, gtn.forward_score, g)

        # Throws on internal self-loop
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, 0)
        g.add_arc(1, 2, 0)
        g.add_arc(1, 1, 0)
        self.assertRaises(ValueError, gtn.forward_score, g)

        # Throws on self-loop in accept node
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, 0)
        g.add_arc(1, 2, 0)
        g.add_arc(2, 2, 0)
        self.assertRaises(ValueError, gtn.forward_score, g)

        # Throws on cycle
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, 0)
        g.add_arc(1, 2, 0)
        g.add_arc(2, 0, 0)
        self.assertRaises(ValueError, gtn.forward_score, g)

        # Throws if a non-start node has no incoming arcs
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 2, 0)
        g.add_arc(1, 2, 0)
        self.assertRaises(ValueError, gtn.forward_score, g)

        # Handles negative infinity
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, -math.inf)
        g.add_arc(0, 1, 1, 1, -math.inf)
        self.assertEqual(gtn.forward_score(g).item(), -math.inf)

        # Handles infinity
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, math.inf)
        g.add_arc(0, 1, 1, 1, 0)
        self.assertEqual(gtn.forward_score(g).item(), math.inf)

        # Single Node
        g = gtn.Graph()
        g.add_node(True, True)
        self.assertEqual(gtn.forward_score(g).item(), 0.0)

        # A simple test case
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, 1)
        g.add_arc(0, 1, 1, 1, 2)
        g.add_arc(0, 1, 2, 2, 3)
        g.add_arc(1, 2, 0, 0, 1)
        g.add_arc(1, 2, 1, 1, 2)
        g.add_arc(1, 2, 2, 2, 3)
        self.assertAlmostEqual(gtn.forward_score(g).item(), (6.8152), places=4)

        # Handle two start nodes
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, -5)
        g.add_arc(0, 2, 0, 0, 1)
        g.add_arc(1, 2, 0, 0, 2)
        expected = math.log(math.exp(1) + math.exp(-5 + 2) + math.exp(2))
        self.assertAlmostEqual(gtn.forward_score(g).item(), (expected), places=4)

        # Handle two accept nodes
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, 2)
        g.add_arc(0, 2, 0, 0, 2)
        g.add_arc(1, 2, 0, 0, 2)
        expected = math.log(2 * math.exp(2) + math.exp(4))
        self.assertAlmostEqual(gtn.forward_score(g).item(), (expected), places=4)

        # Handle case where some arcs don't lead to accepting states
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, False)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, 2)
        g.add_arc(0, 2, 0, 0, 2)
        self.assertEqual(gtn.forward_score(g).item(), 2.0)

        # A more complex test case
        g_str = [
            "0 1",
            "3 4",
            "0 1 0 0 2",
            "0 2 1 1 1",
            "1 2 0 0 2",
            "2 3 0 0 1",
            "2 3 1 1 1",
            "1 4 0 0 2",
            "2 4 1 1 3",
            "3 4 0 0 2",
        ]
        g = create_graph_from_text(g_str)
        self.assertAlmostEqual(gtn.forward_score(g).item(), (8.36931), places=4)

    def test_viterbi_score(self):

        # Check score of empty graph
        g = gtn.Graph()
        self.assertEqual(gtn.viterbi_score(g).item(), -math.inf)

        # A simple test case
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, 1)
        g.add_arc(0, 1, 1, 1, 2)
        g.add_arc(0, 1, 2, 2, 3)
        g.add_arc(1, 2, 0, 0, 1)
        g.add_arc(1, 2, 1, 1, 2)
        g.add_arc(1, 2, 2, 2, 3)
        self.assertEqual(gtn.viterbi_score(g).item(), 6.0)

        # Handle two start nodes
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, -5)
        g.add_arc(0, 2, 0, 0, 1)
        g.add_arc(1, 2, 0, 0, 2)
        self.assertEqual(gtn.viterbi_score(g).item(), 2.0)

        # Handle two accept nodes
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, 2)
        g.add_arc(0, 2, 0, 0, 2)
        g.add_arc(1, 2, 0, 0, 2)
        self.assertEqual(gtn.viterbi_score(g).item(), 4.0)

        # A more complex test case
        g_str = [
            "0 1",
            "3 4",
            "0 1 0 0 2",
            "0 2 1 1 1",
            "1 2 0 0 2",
            "2 3 0 0 1",
            "2 3 1 1 1",
            "1 4 0 0 2",
            "2 4 1 1 3",
            "3 4 0 0 2",
        ]
        g = create_graph_from_text(g_str)
        self.assertEqual(gtn.viterbi_score(g).item(), 7.0)

    def test_viterbi_path(self):

        g = gtn.Graph()

        # Empty graph gives empty path
        self.assertTrue(gtn.equal(gtn.viterbi_path(g), g))

        # Accepting empty string
        g.add_node(True, True)
        self.assertTrue(gtn.equal(gtn.viterbi_path(g), g))

        # A simple test case
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, 1)
        g.add_arc(0, 1, 1, 1, 2)
        g.add_arc(0, 1, 2, 2, 3)
        g.add_arc(1, 2, 0, 0, 1)
        g.add_arc(1, 2, 1, 1, 2)
        g.add_arc(1, 2, 2, 2, 3)

        best = gtn.Graph()
        best.add_node(True)
        best.add_node()
        best.add_node(False, True)
        best.add_arc(0, 1, 2, 2, 3)
        best.add_arc(1, 2, 2, 2, 3)

        path = gtn.viterbi_path(g)
        self.assertTrue(gtn.rand_equivalent(path, best))
        self.assertEqual(gtn.viterbi_score(path).item(), gtn.viterbi_score(g).item())

        # Handle a single node.
        g = gtn.Graph()
        g.add_node(True, True)

        best = gtn.Graph()
        best.add_node(True, True)
        path = gtn.viterbi_path(g)
        self.assertTrue(gtn.rand_equivalent(path, best))
        self.assertEqual(gtn.viterbi_score(path).item(), gtn.viterbi_score(g).item())

        # Handle two start nodes
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, -5)
        g.add_arc(0, 2, 0, 0, 1)
        g.add_arc(1, 2, 0, 0, 2)

        best = gtn.Graph()
        best.add_node(True)
        best.add_node(False, True)
        best.add_arc(0, 1, 0, 0, 2)

        path = gtn.viterbi_path(g)
        self.assertTrue(gtn.rand_equivalent(path, best))
        self.assertEqual(gtn.viterbi_score(path).item(), gtn.viterbi_score(g).item())

        # Handle two accept nodes
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, 3)
        g.add_arc(0, 2, 0, 0, 2)
        g.add_arc(1, 2, 0, 0, 2)

        best = gtn.Graph()
        best.add_node(True)
        best.add_node()
        best.add_node(False, True)
        best.add_arc(0, 1, 0, 0, 3)
        best.add_arc(1, 2, 0, 0, 2)

        path = gtn.viterbi_path(g)
        self.assertTrue(gtn.rand_equivalent(path, best))
        self.assertEqual(gtn.viterbi_score(path).item(), gtn.viterbi_score(g).item())

        # A more complex test case
        g_str = [
            "0 1",
            "3 4",
            "0 1 0 0 2",
            "0 2 1 1 1",
            "1 2 0 0 2",
            "2 3 0 0 1",
            "2 3 1 1 1",
            "1 4 0 0 2",
            "2 4 1 1 3",
            "3 4 0 0 2",
        ]
        g = create_graph_from_text(g_str)

        # There are three options for the best path, the
        # viterbiPath may return any of them.
        best1 = gtn.Graph()
        best1.add_node(True)
        best1.add_node()
        best1.add_node()
        best1.add_node()
        best1.add_node(False, True)
        best1.add_arc(0, 1, 0, 0, 2)
        best1.add_arc(1, 2, 0, 0, 2)
        best1.add_arc(2, 3, 0, 0, 1)
        best1.add_arc(3, 4, 0, 0, 2)

        best2 = gtn.Graph()
        best2.add_node(True)
        best2.add_node()
        best2.add_node()
        best2.add_node()
        best2.add_node(False, True)
        best2.add_arc(0, 1, 0, 0, 2)
        best2.add_arc(1, 2, 0, 0, 2)
        best2.add_arc(2, 3, 1, 1, 1)
        best2.add_arc(3, 4, 0, 0, 2)

        best3 = gtn.Graph()
        best3.add_node(True)
        best3.add_node()
        best3.add_node()
        best3.add_node(False, True)
        best3.add_arc(0, 1, 0, 0, 2)
        best3.add_arc(1, 2, 0, 0, 2)
        best3.add_arc(2, 3, 1, 1, 3)

        path = gtn.viterbi_path(g)
        self.assertTrue(
            (
                gtn.rand_equivalent(path, best1)
                or gtn.rand_equivalent(path, best2)
                or gtn.rand_equivalent(path, best3)
            )
        )

        self.assertEqual(gtn.viterbi_score(path).item(), gtn.viterbi_score(g).item())

    def test_epsilon_composition(self):

        # Simple test case for output epsilon on first graph
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 0, 0, gtn.epsilon, 1.0)
        g1.add_arc(0, 1, 1, 2)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 2, 3)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node(False, True)
        expected.add_arc(0, 0, 0, gtn.epsilon, 1.0)
        expected.add_arc(0, 1, 1, 3)

        self.assertTrue(gtn.equal(gtn.compose(g1, g2), expected))

        # Simple test case for input epsilon on second graph
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 1, 2)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 2, 3)
        g2.add_arc(1, 1, gtn.epsilon, 0, 2.0)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node(False, True)
        expected.add_arc(0, 1, 1, 3)
        expected.add_arc(1, 1, gtn.epsilon, 0, 2.0)

        self.assertTrue(gtn.equal(gtn.compose(g1, g2), expected))

        # This test case is taken from "Weighted Automata Algorithms", Mehryar
        # Mohri, https://cs.nyu.edu/~mohri/pub/hwa.pdf Section 5.1, Figure 7
        symbols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node()
        g1.add_node()
        g1.add_node()
        g1.add_node(False, True)
        g1.add_arc(0, 1, symbols["a"], symbols["a"])
        g1.add_arc(1, 2, symbols["b"], gtn.epsilon)
        g1.add_arc(2, 3, symbols["c"], gtn.epsilon)
        g1.add_arc(3, 4, symbols["d"], symbols["d"])

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node()
        g2.add_node()
        g2.add_node(False, True)
        g2.add_arc(0, 1, symbols["a"], symbols["d"])
        g2.add_arc(1, 2, gtn.epsilon, symbols["e"])
        g2.add_arc(2, 3, symbols["d"], symbols["a"])

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node()
        expected.add_node()
        expected.add_node()
        expected.add_node(False, True)
        expected.add_arc(0, 1, symbols["a"], symbols["d"])
        expected.add_arc(1, 2, symbols["b"], symbols["e"])
        expected.add_arc(2, 3, symbols["c"], gtn.epsilon)
        expected.add_arc(3, 4, symbols["d"], symbols["a"])

        self.assertTrue(gtn.rand_equivalent(gtn.compose(g1, g2), expected))

        # Test multiple input/output epsilon transitions per node
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 0, 1, gtn.epsilon, 1.1)
        g1.add_arc(0, 1, 2, gtn.epsilon, 2.1)
        g1.add_arc(0, 1, 3, gtn.epsilon, 3.1)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, gtn.epsilon, 3, 2.1)
        g2.add_arc(0, 1, 1, 2)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node(False, True)
        expected.add_arc(0, 0, 1, gtn.epsilon, 1.1)
        expected.add_arc(0, 1, 2, 3, 4.2)
        expected.add_arc(0, 1, 3, 3, 5.2)

        self.assertTrue(gtn.rand_equivalent(gtn.compose(g1, g2), expected))

    def test_concat(self):

        # Empty string language
        g = gtn.Graph()
        g.add_node(True, True)

        self.assertTrue(gtn.equal(gtn.concat([]), g))
        self.assertTrue(gtn.rand_equivalent(gtn.concat([g, g]), g))
        self.assertTrue(gtn.rand_equivalent(gtn.concat([g, g, g]), g))

        # Singleton
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, 1)
        self.assertTrue(gtn.equal(gtn.concat([g]), g))

        # Empty language
        g = gtn.Graph()
        g.add_node()
        self.assertTrue(gtn.rand_equivalent(gtn.concat([g, g]), gtn.Graph()))
        self.assertTrue(gtn.rand_equivalent(gtn.concat([g, g, g]), gtn.Graph()))

        # Concat 0 and 1 to get 01
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 1)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node()
        expected.add_node(False, True)
        expected.add_arc(0, 1, 0)
        expected.add_arc(1, 2, 1)

        self.assertTrue(gtn.rand_equivalent(gtn.concat([g1, g2]), expected))

        # Concat 0, 1 and 2, 3 to get 02, 03, 12, 13
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0)
        g1.add_arc(0, 2, 1)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 2, 2)
        g2.add_arc(1, 2, 3)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node()
        expected.add_node(False, True)
        expected.add_arc(0, 1, 0)
        expected.add_arc(0, 1, 1)
        expected.add_arc(1, 2, 2)
        expected.add_arc(1, 2, 3)

        self.assertTrue(gtn.rand_equivalent(gtn.concat([g1, g2]), expected))

    def test_closure(self):

        # Empty graph
        expected = gtn.Graph()
        expected.add_node(True, True)
        self.assertTrue(gtn.equal(gtn.closure(gtn.Graph()), expected))

        # Multi-start, multi-accept
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(True)
        g.add_node(False, True)
        g.add_node(False, True)
        g.add_arc(0, 2, 0, 0, 0.0)
        g.add_arc(0, 3, 1, 2, 2.1)
        g.add_arc(1, 2, 0, 1, 1.0)
        g.add_arc(1, 3, 1, 3, 3.1)

        expected = gtn.Graph()
        expected.add_node(True, True)
        expected.add_node()
        expected.add_node()
        expected.add_node(False, True)
        expected.add_node(False, True)
        expected.add_arc(0, 1, gtn.epsilon)
        expected.add_arc(0, 2, gtn.epsilon)
        expected.add_arc(1, 3, 0, 0, 0.0)
        expected.add_arc(1, 4, 1, 2, 2.1)
        expected.add_arc(2, 3, 0, 1, 1.0)
        expected.add_arc(2, 4, 1, 3, 3.1)
        expected.add_arc(3, 1, gtn.epsilon)
        expected.add_arc(3, 2, gtn.epsilon)
        expected.add_arc(4, 1, gtn.epsilon)
        expected.add_arc(4, 2, gtn.epsilon)

        self.assertTrue(gtn.rand_equivalent(gtn.closure(g), expected))

    def test_sum(self):

        # Empty graph
        self.assertTrue(gtn.equal(gtn.union([]), gtn.Graph()))

        # Check single graph is a no-op
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 1)
        self.assertTrue(gtn.equal(gtn.union([g1]), g1))

        # Simple union
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 1)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node(True)
        expected.add_node(False, True)
        expected.add_node(False, True)
        expected.add_arc(0, 2, 1)
        expected.add_arc(1, 3, 0)
        self.assertTrue(gtn.isomorphic(gtn.union([g1, g2]), expected))

        # Check adding with an empty graph works
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 1)

        g2 = gtn.Graph()

        g3 = gtn.Graph()
        g3.add_node(True, True)
        g3.add_arc(0, 0, 2)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node(False, True)
        expected.add_node(True, True)
        expected.add_arc(0, 1, 1)
        expected.add_arc(2, 2, 2)
        self.assertTrue(gtn.isomorphic(gtn.union([g1, g2, g3]), expected))

    def test_remove(self):
        g = gtn.Graph(False)
        g.add_node(True)
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, gtn.epsilon)
        g.add_arc(1, 2, 0)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node(False, True)
        expected.add_arc(0, 1, 0)
        self.assertTrue(gtn.equal(gtn.remove(g, gtn.epsilon), expected))

        # Check gradient status propagates correctly
        self.assertFalse(gtn.remove(g, gtn.epsilon).calc_grad)

        # Removing other labels works
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, 2, 1)
        g.add_arc(1, 2, 0, 1)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node(False, True)
        expected.add_arc(0, 1, 0, 1)
        self.assertTrue(gtn.equal(gtn.remove(g, 2, 1), expected))

        # No-op on graph without epsilons
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 1)
        g.add_arc(0, 1, 1, 1)
        self.assertTrue(gtn.equal(gtn.remove(g), g))

        # Epsilon only transitions into accepting state
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, 0)
        g.add_arc(0, 2, 1)
        g.add_arc(1, 3, gtn.epsilon)
        g.add_arc(2, 3, gtn.epsilon)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node(False, True)
        expected.add_node(False, True)
        expected.add_arc(0, 1, 0)
        expected.add_arc(0, 2, 1)
        self.assertTrue(gtn.equal(gtn.remove(g), expected))

        # Only remove an arc, no removed nodes
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, gtn.epsilon)
        g.add_arc(0, 2, 1)
        g.add_arc(2, 1, 0)
        g.add_arc(1, 3, 1)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node()
        expected.add_node()
        expected.add_node(False, True)
        expected.add_arc(0, 2, 1)
        expected.add_arc(2, 1, 0)
        expected.add_arc(1, 3, 1)
        expected.add_arc(0, 3, 1)
        self.assertTrue(gtn.equal(gtn.remove(g), expected))

        # Successive epsilons
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node()
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, 0)
        g.add_arc(1, 2, gtn.epsilon)
        g.add_arc(2, 3, gtn.epsilon)
        g.add_arc(2, 4, 1)
        g.add_arc(3, 4, 2)
        g.add_arc(1, 4, 0)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node()
        expected.add_node(False, True)
        expected.add_arc(0, 1, 0)
        expected.add_arc(1, 2, 0)
        expected.add_arc(1, 2, 1)
        expected.add_arc(1, 2, 2)
        self.assertTrue(gtn.equal(gtn.remove(g), expected))

        # Multiple interior removals

        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node()
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 1, gtn.epsilon)
        g.add_arc(1, 2, gtn.epsilon)
        g.add_arc(2, 3, 0)
        g.add_arc(3, 4, 0)

        expected = gtn.Graph()
        expected.add_node(True)
        expected.add_node()
        expected.add_node(False, True)
        expected.add_arc(0, 1, 0)
        expected.add_arc(1, 2, 0)
        self.assertTrue(gtn.equal(gtn.remove(g), expected))


if __name__ == "__main__":
    unittest.main()
