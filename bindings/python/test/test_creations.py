"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import unittest
import random
import gtn


class CreationsTestCase(unittest.TestCase):
    def test_scalar_creation(self):
        weight = random.random()
        g = gtn.scalar_graph(weight, False)
        self.assertEqual(g.num_arcs(), 1)
        self.assertEqual(g.labels_to_list(), [gtn.epsilon])
        self.assertEqual(g.num_nodes(), 2)
        self.assertEqual(len(g.weights_to_list()), 1)
        self.assertAlmostEqual(g.weights_to_list()[0], weight, places=5)
        self.assertAlmostEqual(g.item(), weight, places=5)
        self.assertFalse(g.calc_grad)

    def test_linear_creation(self):
        M = 5
        N = 10
        arr = [random.random() for _ in range(M * N)]
        g = gtn.linear_graph(M, N)
        g.set_weights(arr)
        self.assertEqual(g.num_nodes(), M + 1)
        self.assertEqual(g.num_arcs(), M * N)

        self.assertEqual(g.labels_to_list(), [j for _ in range(M) for j in range(N)])
        weights = g.weights_to_list()
        for i, w in enumerate(weights):
            self.assertAlmostEqual(w, arr[i], places=5)


if __name__ == "__main__":
    unittest.main()
