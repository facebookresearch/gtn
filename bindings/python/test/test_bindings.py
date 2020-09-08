#!/usr/bin/env python3


import ctypes
import numpy as np
import struct
import unittest
import random
import tempfile

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
        g.add_arc(src_node=1, dst_node=2, label=0)
        g.add_arc(1, 1, ilabel=1, olabel=2, weight=2.1)
        g.add_arc(2, 3, 2)
        self.g = g

    def test_graph_basic(self):
        self.assertEqual(self.g.num_nodes(), 5)
        self.assertEqual(self.g.num_start(), 1)
        self.assertEqual(self.g.num_accept(), 1)

        self.assertEqual(self.g.num_arcs(), 5)
        self.assertEqual(self.g.labels_to_list(), [0, 1, 0, 1, 2])
        self.assertEqual(self.g.labels_to_list(False), [0, 1, 0, 2, 2])

    def test_graph_weights_get(self):
        weights = self.g.weights()
        weights_list = self.g.weights_to_list()
        weights_numpy = self.g.weights_to_numpy()
        expected = [0, 0, 0, 2.1, 0]
        # get weights as ptr
        length = 5
        get_weights_numpy = np.frombuffer(
            (ctypes.c_float * length).from_address(weights), np.float32
        )
        self.assertListAlmostEqual(get_weights_numpy.tolist(), expected, places=4)

        # get weights as list
        self.assertListAlmostEqual(weights_list, expected, places=4)

        # get weights as numpy
        self.assertListAlmostEqual(weights_numpy.tolist(), expected, places=4)

    def test_graph_weights_set(self):
        weights_original = self.g.weights()
        weights_new_expected = [1.1, -3.4, 0, 0.5, 0]

        # set weights as list
        self.g.set_weights(weights_new_expected)
        weights_new = self.g.weights_to_list()
        self.assertListAlmostEqual(weights_new, weights_new_expected, places=4)
        self.g.set_weights(weights_original)

        # set weights via numpy
        weights_new_arr = np.array(weights_new_expected, dtype="f")
        self.g.set_weights(weights_new_arr)
        weights_new = self.g.weights_to_numpy()
        self.assertListAlmostEqual(
            weights_new.tolist(), weights_new_arr.tolist(), places=4
        )
        self.g.set_weights(weights_original)

        # set weights via ptr
        weights_new_arr_ptr = weights_new_arr.__array_interface__["data"][0]
        self.g.set_weights(weights_new_arr_ptr)
        weights_new = self.g.weights_to_list()
        self.assertListAlmostEqual(weights_new, weights_new_arr.tolist(), places=4)
        self.g.set_weights(weights_original)

    def test_comparisons(self):
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0)

        self.assertTrue(gtn.equal(g1, g2))
        self.assertTrue(gtn.isomorphic(g1, g2))

        g2 = gtn.Graph()
        g2.add_node(False, True)
        g2.add_node(True)
        g2.add_arc(1, 0, 0)

        self.assertFalse(gtn.equal(g1, g2))
        self.assertTrue(gtn.isomorphic(g1, g2))

    def test_save_load(self):
        with tempfile.NamedTemporaryFile(mode="r") as fid:
            gtn.savetxt(fid.name, self.g)
            loaded = gtn.loadtxt(fid.name)
            self.assertTrue(gtn.isomorphic(self.g, loaded))

        with tempfile.NamedTemporaryFile(mode="r") as fid:
            gtn.save(fid.name, self.g)
            loaded = gtn.load(fid.name)
            self.assertTrue(gtn.isomorphic(self.g, loaded))

    def test_scalar_graph(self):
        weight = random.random()
        g = gtn.scalar_graph(weight)
        self.assertListAlmostEqual(g.weights_to_list(), [weight])
        self.assertEqual(g.num_arcs(), 1)
        self.assertEqual(g.num_nodes(), 2)

        g_epsilon = gtn.scalar_graph()
        self.assertEqual(g_epsilon.item(), gtn.epsilon)


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


class ParallelTestCase(GTNModuleTestCase):
    def test_parallel_one_arg(self):
        inputs = [gtn.scalar_graph(k) for k in [1.0, 2.0, 3.0]]
        outputs = gtn.negate(inputs)

        expected = []
        for g in inputs:
            expected.append(gtn.negate(g))

        self.assertEqual(len(outputs), len(inputs))
        for i in range(0, len(expected)):
            self.assertTrue(gtn.equal(outputs[i], expected[i]))

    def test_parallel_two_arg(self):
        inputs1 = [gtn.scalar_graph(k) for k in [1.0, 2.0, 3.0]]
        inputs2 = [gtn.scalar_graph(k) for k in [1.0, 2.0, 3.0]]
        outputs = gtn.add(inputs1, inputs2)

        expected = []
        for g1, g2 in zip(inputs1, inputs2):
            expected.append(gtn.add(g1, g2))

        self.assertEqual(len(outputs), len(inputs1), len(inputs2))
        for i in range(0, len(expected)):
            self.assertTrue(gtn.equal(outputs[i], expected[i]))

    def test_parallel_vector_arg(self):
        inputList = [
            gtn.scalar_graph(1.0),
            gtn.scalar_graph(2.0),
            gtn.scalar_graph(3.0),
        ]
        inputs = [inputList, inputList, inputList]
        outputs = gtn.concat(inputs)

        expected = []
        for gList in inputs:
            expected.append(gtn.concat(gList))

        self.assertEqual(len(outputs), len(inputs))
        for i in range(0, len(expected)):
            self.assertTrue(gtn.equal(outputs[i], expected[i]))

    def test_backward_calls_once(self):
        g1 = gtn.scalar_graph(1)
        g2 = gtn.scalar_graph(1)
        gout = gtn.add(g1, g2)
        gtn.backward([gout])
        pmap_grad = gout.grad()
        gout = gtn.add(g1, g2)
        gtn.backward(gout)
        grad = gout.grad()
        self.assertTrue(gtn.equal(pmap_grad, grad))

    def test_parallel_backward(self):
        inputs1 = [gtn.scalar_graph(k) for k in [1.0, 2.0, 3.0]]
        inputs2 = [gtn.scalar_graph(k) for k in [1.0, 2.0, 3.0]]

        outputs = gtn.add(inputs1, inputs2)
        gtn.backward(outputs)
        # Test gradients
        inputs1 = [gtn.scalar_graph(k) for k in [1.0, 2.0, 3.0]]
        inputs2 = [gtn.scalar_graph(k) for k in [1.0, 2.0, 3.0]]
        outputs = gtn.add(inputs1, inputs2)
        gradIn = gtn.scalar_graph(5.0)
        gtn.backward(outputs, [gradIn], [False])

        inputs1Dup = [gtn.scalar_graph(k) for k in [1.0, 2.0, 3.0]]
        inputs2Dup = [gtn.scalar_graph(k) for k in [1.0, 2.0, 3.0]]
        expected = []
        for g1, g2 in zip(inputs1Dup, inputs2Dup):
            expected.append(gtn.add(g1, g2))
        for g in expected:
            gtn.backward(g, gtn.scalar_graph(5.0))

        for i in range(0, len(expected)):
            self.assertTrue(gtn.equal(inputs1[i].grad(), inputs1Dup[i].grad()))
            self.assertTrue(gtn.equal(inputs2[i].grad(), inputs2Dup[i].grad()))

    def test_parallel_func(self):
        B = 3
        inputs1 = [gtn.scalar_graph(k) for k in [1.0, 2.0, 3.0]]
        inputs2 = [gtn.scalar_graph(k) for k in [1.0, 2.0, 3.0]]

        out = [None] * B

        def process(b):
            out[b] = gtn.add(gtn.add(inputs1[b], inputs1[b]), gtn.negate(inputs2[b]))

        gtn.parallel_for(process, range(B))

        expected = []
        for b in range(B):
            expected.append(
                gtn.add(gtn.add(inputs1[b], inputs1[b]), gtn.negate(inputs2[b]))
            )

        self.assertEqual(len(out), len(expected))
        for i in range(len(expected)):
            self.assertTrue(gtn.equal(out[i], expected[i]))


class AutogradTestCase(GTNModuleTestCase):
    def test_calc_grad(self):
        g1 = gtn.Graph(False)
        g1.calc_grad = True
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 1, 1, 1.0)

        g2 = gtn.Graph(True)
        g2.calc_grad = False
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 0, 1, 1, 1.0)

        result = gtn.add(g1, g2)
        gtn.backward(result)

        self.assertTrue(g1.grad().item() == 1.0)
        with self.assertRaises(RuntimeError):
            g2.grad()

    def test_retain_graph(self):
        # The graph is not retained by default
        g1 = gtn.Graph(True)
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0, 0, 3.0)

        g2 = gtn.Graph(True)
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 0, 3.0)

        result = gtn.add(g1, g2)
        gtn.backward(result)
        with self.assertRaises(ValueError):
            gtn.backward(result)

        # Check the graph is retained
        g1.zero_grad()
        g2.zero_grad()
        result = gtn.add(g1, g2)
        gtn.backward(result, True)
        g1.zero_grad()
        g2.zero_grad()
        result.zero_grad()
        gtn.backward(result, True)
        self.assertTrue(g1.grad().item() == 1.0)
        self.assertTrue(g2.grad().item() == 1.0)

    def test_input_grad(self):
        # Check that provided input gradients are used.
        g1 = gtn.Graph(True)
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0, 0, 3.0)

        g2 = gtn.Graph(True)
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 0, 3.0)

        result = gtn.add(g1, g2)

        deltas = gtn.Graph()
        deltas.add_node(True)
        deltas.add_node(False, True)
        deltas.add_arc(0, 1, 0, 0, 7.0)
        gtn.backward(result, deltas)
        self.assertTrue(g1.grad().item() == 7.0)
        self.assertTrue(g2.grad().item() == 7.0)


if __name__ == "__main__":
    unittest.main()
