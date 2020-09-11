import math
import unittest
import gtn

from test_helpers import create_graph_from_text


def isclose(a, b, relTol=1e-5, absTol=1e-3):
    return abs(a - b) <= max(relTol * max(abs(a), abs(b)), absTol)


def numerical_grad_check(func, input, epsilon, relTol):
    # Numerical gradient check.
    gradPass = True
    for a in range(input.num_arcs()):
        weights = input.weights_to_list()
        w_a = weights[a]
        weights[a] = w_a + epsilon
        input.set_weights(weights)
        high = func(input).item()
        weights[a] = w_a - epsilon
        input.set_weights(weights)
        low = func(input).item()
        numgrad = (high - low) / (2 * epsilon)
        g_a = input.grad().weights_to_list()[a]
        gradPass = gradPass and isclose(g_a, numgrad, relTol)
        weights[a] = w_a
        input.set_weights(weights)
    return gradPass


class AutogradTestCase(unittest.TestCase):
    def test_autograd(self):
        # The graph is not retained by default
        g1 = gtn.scalar_graph(3.0)
        g2 = gtn.scalar_graph(3.0)

        result = gtn.add(g1, g2)
        gtn.backward(result)
        # Cannot backward twice when graph is cleared.
        self.assertRaises(ValueError, gtn.backward, result)

        # Check the graph is retained
        g1.zero_grad()
        g2.zero_grad()
        result = gtn.add(g1, g2)
        gtn.backward(result, True)
        result.zero_grad()
        g1.zero_grad()
        g2.zero_grad()
        gtn.backward(result, True)
        self.assertEqual(g1.grad().item(), 1.0)
        self.assertEqual(g2.grad().item(), 1.0)

        # Check that provided input gradients are used.
        g1.zero_grad()
        g2.zero_grad()
        result = gtn.add(g1, g2)
        deltas = gtn.Graph()
        deltas.add_node(True)
        deltas.add_node(False, True)
        deltas.add_arc(0, 1, 0, 0, 7.0)
        gtn.backward(result, deltas)
        self.assertEqual(g1.grad().item(), 7.0)
        self.assertEqual(g2.grad().item(), 7.0)

    def test_scalar_ops_grad(self):
        g1 = gtn.scalar_graph(3.0)

        result = gtn.negate(g1)
        gtn.backward(result)
        self.assertEqual(g1.grad().item(), -1.0)

        g1.zero_grad()

        g2 = gtn.scalar_graph(4.0)

        result = gtn.add(g1, g2)
        gtn.backward(result)
        self.assertEqual(g1.grad().item(), 1.0)
        self.assertEqual(g2.grad().item(), 1.0)

        g1.zero_grad()
        g2.zero_grad()

        result = gtn.subtract(g1, g2)
        gtn.backward(result)
        self.assertEqual(g1.grad().item(), 1.0)
        self.assertEqual(g2.grad().item(), -1.0)
        g1.zero_grad()
        g2.zero_grad()

        result = gtn.add(gtn.add(g1, g2), g1)
        gtn.backward(result)
        self.assertEqual(g1.grad().item(), 2.0)
        self.assertEqual(g2.grad().item(), 1.0)
        g1.zero_grad()

        g2nograd = gtn.scalar_graph(4.0, False)

        result = gtn.add(g1, g2nograd)
        gtn.backward(result)
        self.assertEqual(g1.grad().item(), 1.0)
        self.assertRaises(RuntimeError, g2nograd.grad)

    def test_clone_project_grad(self):
        g1 = gtn.scalar_graph(3.0)
        g2 = gtn.scalar_graph(4.0)

        cloned = gtn.clone(g1)

        result = gtn.add(g1, g2)
        gtn.backward(result)

        # Cloned wasn't used in the computation
        self.assertRaises(RuntimeError, cloned.grad)

        # Cloned was used in the computation
        g1.zero_grad()
        g2.zero_grad()
        result = gtn.add(cloned, g2)
        gtn.backward(result)
        self.assertTrue(gtn.equal(cloned.grad(), g1.grad()))

    def test_compose_grad(self):
        first = gtn.Graph()
        first.add_node(True)
        first.add_node()
        first.add_node()
        first.add_node()
        first.add_node(False, True)
        first.add_arc(0, 1, 0, 0, 0)
        first.add_arc(0, 1, 1, 1, 1)
        first.add_arc(0, 1, 2, 2, 2)
        first.add_arc(1, 2, 0, 0, 0)
        first.add_arc(1, 2, 1, 1, 1)
        first.add_arc(1, 2, 2, 2, 2)
        first.add_arc(2, 3, 0, 0, 0)
        first.add_arc(2, 3, 1, 1, 1)
        first.add_arc(2, 3, 2, 2, 2)
        first.add_arc(3, 4, 0, 0, 0)
        first.add_arc(3, 4, 1, 1, 1)
        first.add_arc(3, 4, 2, 2, 2)

        second = gtn.Graph()
        second.add_node(True)
        second.add_node()
        second.add_node(False, True)
        second.add_arc(0, 1, 0, 0, 3.5)
        second.add_arc(1, 1, 0, 0, 2.5)
        second.add_arc(1, 2, 1, 1, 1.5)
        second.add_arc(2, 2, 1, 1, 4.5)

        composed = gtn.compose(first, second)
        gtn.backward(composed)

        gradsFirst = [1, 0, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0]
        gradsSecond = [1, 2, 3, 2]

        self.assertEqual(gradsFirst, first.grad().weights_to_list())
        self.assertEqual(gradsSecond, second.grad().weights_to_list())

    def test_grad_available(self):
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
        self.assertFalse(g.is_grad_available())
        gtn.backward(gtn.forward_score(g))
        self.assertTrue(g.is_grad_available())

    def test_forward_score_grad(self):
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
        gtn.backward(gtn.forward_score(g))
        self.assertTrue(numerical_grad_check(gtn.forward_score, g, 1e-3, 1e-3))

        # Handle two start nodes
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, -5)
        g.add_arc(0, 2, 0, 0, 1)
        g.add_arc(1, 2, 0, 0, 2)
        gtn.backward(gtn.forward_score(g))
        self.assertTrue(numerical_grad_check(gtn.forward_score, g, 1e-3, 1e-3))

        denom = 1 / (math.exp(-3) + math.exp(1) + math.exp(2))
        grad = g.grad()
        grad_weights = grad.weights_to_list()
        self.assertAlmostEqual(grad_weights[0], (denom * math.exp(-3)))
        self.assertAlmostEqual(grad_weights[1], (denom * math.exp(1)))
        self.assertAlmostEqual(grad_weights[2], (denom * (math.exp(-3) + math.exp(2))))

        # Handle two accept nodes
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, 2)
        g.add_arc(0, 2, 0, 0, 2)
        g.add_arc(1, 2, 0, 0, 2)
        gtn.backward(gtn.forward_score(g))
        self.assertTrue(numerical_grad_check(gtn.forward_score, g, 1e-3, 1e-3))

        denom = 1 / (2 * math.exp(2) + math.exp(4))
        grad = g.grad()
        grad_weights = grad.weights_to_list()
        self.assertAlmostEqual(grad_weights[0], (denom * (math.exp(2) + math.exp(4))))
        self.assertAlmostEqual(grad_weights[1], (denom * math.exp(2)))
        self.assertAlmostEqual(grad_weights[2], (denom * math.exp(4)))

        # Handle case where some arcs don't lead to accepting states
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, False)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, 2)
        g.add_arc(0, 2, 0, 0, 2)
        gtn.backward(gtn.forward_score(g))
        self.assertTrue(numerical_grad_check(gtn.forward_score, g, 1e-3, 1e-3))
        grad = g.grad()
        grad_weights = grad.weights_to_list()
        self.assertAlmostEqual(grad_weights[0], (0.0))
        self.assertAlmostEqual(grad_weights[1], (1.0))

        # Handles negative infinity
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, -math.inf)
        g.add_arc(0, 1, 1, 1, -math.inf)
        gtn.backward(gtn.forward_score(g))

        grad = g.grad()
        grad_weights = grad.weights_to_list()
        self.assertTrue(math.isnan(grad_weights[0]))
        self.assertTrue(math.isnan(grad_weights[1]))

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 0, -math.inf)
        g2.add_arc(0, 1, 1, 1, 1.0)
        gtn.backward(gtn.forward_score(g2))

        grad2 = g2.grad()
        grad_weights = grad2.weights_to_list()
        self.assertAlmostEqual(grad_weights[0], (0.0))
        self.assertAlmostEqual(grad_weights[1], (1.0))

        # Handles infinity
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, math.inf)
        g.add_arc(0, 1, 1, 1, math.inf)
        gtn.backward(gtn.forward_score(g))
        grad = g.grad()
        grad_weights = grad.weights_to_list()
        self.assertTrue(math.isnan(grad_weights[0]))
        self.assertTrue(math.isnan(grad_weights[1]))

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 0, math.inf)
        g2.add_arc(0, 1, 1, 1, 1.0)
        gtn.backward(gtn.forward_score(g2))
        grad2 = g2.grad()
        grad_weights = grad.weights_to_list()
        self.assertTrue(math.isnan(grad_weights[0]))
        self.assertTrue(math.isnan(grad_weights[1]))

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
        gtn.backward(gtn.forward_score(g))
        self.assertTrue(numerical_grad_check(gtn.forward_score, g, 1e-3, 1e-3))

    def test_viterbi_score_grad(self):
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
        gtn.backward(gtn.viterbi_score(g))
        expected = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        self.assertEqual(g.grad().weights_to_list(), expected)

        # Handle two start nodes
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, -5)
        g.add_arc(0, 2, 0, 0, 1)
        g.add_arc(1, 2, 0, 0, 2)
        gtn.backward(gtn.viterbi_score(g))
        expected = [0.0, 0.0, 1.0]
        self.assertEqual(g.grad().weights_to_list(), expected)

        # Handle two accept nodes
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        g.add_node(False, True)
        g.add_arc(0, 1, 0, 0, 2)
        g.add_arc(0, 2, 0, 0, 2)
        g.add_arc(1, 2, 0, 0, 2)
        gtn.backward(gtn.viterbi_score(g))
        expected = [1.0, 0.0, 1.0]
        self.assertEqual(g.grad().weights_to_list(), expected)

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
        gtn.backward(gtn.viterbi_score(g))
        # two possible paths with same viterbi score
        expected1 = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        expected2 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        self.assertTrue(
            g.grad().weights_to_list() == expected1
            or g.grad().weights_to_list() == expected2
        )

    def test_viterbi_path_grad(self):
        g_str = [
            "0 1",
            "3 4",
            "0 1 0 0 2",
            "0 2 1 1 1",
            "1 2 0 0 2",
            "2 3 0 0 1",
            "2 3 1 1 3",
            "1 4 0 0 2",
            "2 4 1 1 3",
            "3 4 0 0 2",
        ]
        g = create_graph_from_text(g_str)
        gtn.backward(gtn.viterbi_path(g))
        expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        self.assertEqual(g.grad().weights_to_list(), expected)
        g.zero_grad()

        def forward_fn(g):
            paths = [gtn.viterbi_path(g), gtn.viterbi_path(g), gtn.viterbi_path(g)]
            return gtn.forward_score(gtn.union(paths))

        gtn.backward(forward_fn(g))
        self.assertTrue(numerical_grad_check(forward_fn, g, 1e-3, 1e-5))

    def test_sample_grad(self):
        g = gtn.Graph()
        g.add_node(True)
        g.add_node()
        g.add_node(False, True)
        g.add_arc(0, 0, 0)
        g.add_arc(0, 1, 1)
        g.add_arc(1, 0, 2)
        g.add_arc(1, 2, 3)

        for i in range(5):
            g.zero_grad()
            path = gtn.sample(g)
            # One for each arc in the original graph
            grads = [0.0, 0.0, 0.0, 0.0]
            path_labels = path.labels_to_list()
            for a in range(path.num_arcs()):
                grads[path_labels[a]] += 1

            gtn.backward(path)
            self.assertTrue(grads == g.grad().weights_to_list())

        def test_sum_grad(self):
            g1 = gtn.Graph()
            g1.add_node(True)
            g1.add_node()
            g1.add_node(False, True)
            g1.add_arc(0, 1, 0)
            g1.add_arc(1, 2, 1)

            # Works with a no gradient graph
            g2 = gtn.Graph()(False)
            g2.add_node(True)
            g2.add_node()
            g2.add_node(False, True)
            g2.add_arc(0, 1, 0)
            g2.add_arc(1, 2, 1)

            g3 = gtn.Graph()
            g3.add_node(True)
            g3.add_node()
            g3.add_node(False, True)
            g3.add_arc(0, 1, 0)
            g3.add_arc(1, 2, 1)

            gtn.backward(gtn.forward_score(gtn.union([g1, g2, g3])))

            def forward_fn1(g, g2=g2, g3=g3):
                return gtn.forward_score(gtn.union([g, g2, g3]))

            self.assertTrue(numerical_grad_check(forward_fn1, g1, 1e-4, 1e-3))

            def forward_fn2(g, g1=g1, g2=g2):
                return gtn.forward_score(gtn.union([g1, g2, g]))

            self.assertTrue(numerical_grad_check(forward_fn2, g3, 1e-4, 1e-3))

            CHECK_THROWS(g2.grad())

        def test_concat_grad(self):
            g1 = gtn.Graph()
            g1.add_node(True)
            g1.add_node()
            g1.add_node(False, True)
            g1.add_arc(0, 1, 0)
            g1.add_arc(1, 2, 1)

            # Works with a no gradient graph
            g2 = gtn.Graph()(False)
            g2.add_node(True)
            g2.add_node()
            g2.add_node(False, True)
            g2.add_arc(0, 1, 0)
            g2.add_arc(1, 2, 1)

            g3 = gtn.Graph()
            g3.add_node(True)
            g3.add_node()
            g3.add_node(False, True)
            g3.add_arc(0, 1, 0)
            g3.add_arc(1, 2, 1)

            gtn.backward(gtn.forward_score(concat([g1, g2, g3])))

            def forward_fn1(g, g2=g2, g3=g3):
                return gtn.forward_score(gtn.concat([g, g2, g3]))

            self.assertTrue(numerical_grad_check(forward_fn1, g1, 1e-4, 1e-3))

            def forward_fn2(g, g1=g1, g2=g2):
                return gtn.forward_score(gtn.concat([g1, g2, g]))

            self.assertTrue(numerical_grad_check(forward_fn2, g1, 1e-4, 1e-3))

            CHECK_THROWS(g2.grad())

        def test_closure_grad(self):
            g1 = gtn.Graph()
            g1.add_node(True)
            g1.add_node(False, True)
            g1.add_arc(0, 1, 0, 0, 1.3)
            g1.add_arc(1, 1, 1, 1, 2.1)

            g2 = gtn.Graph()
            g2.add_node(True)
            g2.add_node()
            g2.add_node()
            g2.add_node()
            g2.add_node(False, True)
            g2.add_arc(0, 1, 0)
            g2.add_arc(0, 1, 1)
            g2.add_arc(1, 2, 0)
            g2.add_arc(1, 2, 1)
            g2.add_arc(2, 3, 0)
            g2.add_arc(2, 3, 1)
            g2.add_arc(3, 4, 0)
            g2.add_arc(3, 4, 1)

            gtn.backward(gtn.forward_score(gtn.compose(closure(g1), g2)))

            def forward_fn(g, g2=g2):
                return gtn.forward_score(gtn.compose(closure(g), g2))

            self.assertTrue(numerical_grad_check(forward_fn, g1, 1e-3, 1e-3))


if __name__ == "__main__":
    unittest.main()
