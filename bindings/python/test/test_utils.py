import tempfile
import gtn
import unittest
from test_helpers import create_graph_from_text


class UtilsTestCase(unittest.TestCase):
    def test_equality(self):
        # Empty graph is equal to itself
        g1 = gtn.Graph()
        g2 = gtn.Graph()
        self.assertTrue(gtn.equal(g1, g2))

        # Different start node
        g1 = gtn.Graph()
        g1.add_node(True)

        g2 = gtn.Graph()
        g2.add_node(False)
        self.assertFalse(gtn.equal(g1, g2))

        # Simple equality
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node()

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node()
        self.assertTrue(gtn.equal(g1, g2))

        # Different arc label
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node()
        g1.add_arc(0, 1, 0)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node()
        g2.add_arc(0, 1, 1)
        self.assertFalse(gtn.equal(g1, g2))

        # Different arc weight
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node()
        g1.add_arc(0, 1, 0, 0, 1.2)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node()
        g2.add_arc(0, 1, 0, 0, 2.2)
        self.assertFalse(gtn.equal(g1, g2))

        # Self loop in g1
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node()
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0)
        g1.add_arc(0, 1, 1)
        g1.add_arc(1, 1, 1)
        g1.add_arc(1, 2, 2)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node()
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0)
        g2.add_arc(0, 1, 1)
        g2.add_arc(1, 2, 2)
        self.assertFalse(gtn.equal(g1, g2))

        # Equals
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node()
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0, 0, 2.1)
        g1.add_arc(0, 1, 1, 1, 3.1)
        g1.add_arc(1, 1, 1, 1, 4.1)
        g1.add_arc(1, 2, 2, 2, 5.1)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node()
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 0, 2.1)
        g2.add_arc(0, 1, 1, 1, 3.1)
        g2.add_arc(1, 1, 1, 1, 4.1)
        g2.add_arc(1, 2, 2, 2, 5.1)
        self.assertTrue(gtn.equal(g1, g2))

        # Different arc order
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node()
        g1.add_node(False, True)
        g1.add_arc(0, 1, 1, 1, 3.1)
        g1.add_arc(0, 1, 0, 0, 2.1)
        g1.add_arc(1, 1, 1, 1, 4.1)
        g1.add_arc(1, 2, 2, 2, 5.1)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node()
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 0, 2.1)
        g2.add_arc(0, 1, 1, 1, 3.1)
        g2.add_arc(1, 2, 2, 2, 5.1)
        g2.add_arc(1, 1, 1, 1, 4.1)
        self.assertTrue(gtn.equal(g1, g2))

        # Repeat arcs
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 1, 1, 3.1)
        g1.add_arc(0, 1, 1, 1, 3.1)
        g1.add_arc(0, 1, 1, 1, 4.1)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 1, 1, 3.1)
        g2.add_arc(0, 1, 1, 1, 4.1)
        g2.add_arc(0, 1, 1, 1, 4.1)
        self.assertFalse(gtn.equal(g1, g2))

        # Transducer with different outputs
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0, 1, 2.1)
        g1.add_arc(1, 1, 1, 3, 4.1)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 1, 2.1)
        g2.add_arc(1, 1, 1, 4, 4.1)
        self.assertFalse(gtn.equal(g1, g2))

    def test_isomorphic(self):

        g1 = gtn.Graph()
        g1.add_node(True)

        g2 = gtn.Graph()
        g2.add_node(True)

        self.assertTrue(gtn.isomorphic(g1, g2))

        g1 = gtn.Graph()
        g1.add_node(True)

        g2 = gtn.Graph()
        g2.add_node(True, True)

        self.assertFalse(gtn.isomorphic(g1, g2))

        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0, 0, 1)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 0, 1)

        self.assertTrue(gtn.isomorphic(g1, g2))

        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0, 0, 1)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 0, 1)
        g2.add_arc(0, 1, 0, 0, 1)

        self.assertFalse(gtn.isomorphic(g1, g2))

        # Graph with loops
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node()
        g1.add_node(False, True)
        g1.add_arc(0, 2, 0)
        g1.add_arc(0, 1, 0)
        g1.add_arc(1, 1, 3)
        g1.add_arc(2, 1, 3)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_node()
        g2.add_arc(0, 1, 0)
        g2.add_arc(1, 2, 3)
        g2.add_arc(0, 2, 0)
        g2.add_arc(2, 2, 3)

        self.assertTrue(gtn.isomorphic(g1, g2))

        # Repeat arcs
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 1, 1, 3.1)
        g1.add_arc(0, 1, 1, 1, 3.1)
        g1.add_arc(0, 1, 1, 1, 4.1)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 1, 1, 3.1)
        g2.add_arc(0, 1, 1, 1, 4.1)
        g2.add_arc(0, 1, 1, 1, 4.1)
        self.assertFalse(gtn.isomorphic(g1, g2))

        # Transducer with different outputs
        g1 = gtn.Graph()
        g1.add_node(True)
        g1.add_node(False, True)
        g1.add_arc(0, 1, 0, 1, 2.1)
        g1.add_arc(1, 1, 1, 3, 4.1)

        g2 = gtn.Graph()
        g2.add_node(True)
        g2.add_node(False, True)
        g2.add_arc(0, 1, 0, 1, 2.1)
        g2.add_arc(1, 1, 1, 4, 4.1)
        self.assertFalse(gtn.isomorphic(g1, g2))

    def test_loadtxt(self):

        g1 = gtn.Graph()
        g1.add_node(True, True)
        g1.add_node(False, True)
        g1.add_node()
        g1.add_arc(0, 0, 1)
        g1.add_arc(0, 2, 1, 1, 1.1)
        g1.add_arc(2, 1, 2, 2, 2.1)

        g_str = ["0", "0 1", "0 0 1 1 0", "0 2 1 1 1.1", "2 1 2 2 2.1"]
        g2 = create_graph_from_text(g_str)
        self.assertTrue(gtn.equal(g1, g2))
        self.assertTrue(gtn.isomorphic(g1, g2))

        _, tmpfile = tempfile.mkstemp()

        # Write the test file
        gtn.savetxt(tmpfile, g2)
        g3 = gtn.loadtxt(tmpfile)
        self.assertTrue(gtn.equal(g1, g3))

        # Empty graph doesn't load

        g_str = [""]
        self.assertRaises(ValueError, create_graph_from_text, g_str)

        # Graph without accept nodes doesn't load

        g_str = ["1"]
        self.assertRaises(ValueError, create_graph_from_text, g_str)

        # Graph with repeat start nodes doesn't load

        g_str = ["1 0 0", "0 1"]
        self.assertRaises(ValueError, create_graph_from_text, g_str)

        # Graph loads if the start and accept nodes are specified

        g_str = ["0", "1"]
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(False, True)
        self.assertTrue(gtn.equal(g, create_graph_from_text(g_str)))

        # Graph doesn't load if arc incorrect

        g_str = ["0", "1", "0 2"]
        self.assertRaises(ValueError, create_graph_from_text, g_str)
        g_str = ["0", "1", "0 1 2 3 4 5"]
        self.assertRaises(ValueError, create_graph_from_text, g_str)

        # Transducer loads

        g1 = gtn.Graph()
        g1.add_node(True, True)
        g1.add_node(False, True)
        g1.add_node()
        g1.add_arc(0, 0, 1, 1)
        g1.add_arc(0, 2, 1, 2, 1.1)
        g1.add_arc(2, 1, 2, 3, 2.1)

        g_str = ["0", "0 1", "0 0 1", "0 2 1 2 1.1", "2 1 2 3 2.1"]
        g2 = create_graph_from_text(g_str)
        self.assertTrue(gtn.equal(g1, g2))
        self.assertTrue(gtn.isomorphic(g1, g2))

    def test_savetxt(self):

        # Acceptor test
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(True)
        g.add_node()
        g.add_node()
        g.add_node(False, True)
        g.add_node(False, True)

        g.add_arc(0, 1, 0, 0, 1.1)
        g.add_arc(1, 2, 1, 1, 2.1)
        g.add_arc(2, 3, 2, 2, 3.1)
        g.add_arc(3, 4, 3, 3, 4.1)
        g.add_arc(4, 5, 4, 4, 5.1)

        g_str = [
            "0 1",
            "4 5",
            "0 1 0 0 1.1",
            "1 2 1 1 2.1",
            "2 3 2 2 3.1",
            "3 4 3 3 4.1",
            "4 5 4 4 5.1",
        ]
        g = create_graph_from_text(g_str)
        _, tmpfile = tempfile.mkstemp()
        gtn.savetxt(tmpfile, g)
        with open(tmpfile) as f:
            lines = f.readlines()
            self.assertEqual(len(g_str), len(lines))
            for i, line in enumerate(lines):
                self.assertEqual(line.strip(), g_str[i])

        # Transducer test
        g = gtn.Graph()
        g.add_node(True)
        g.add_node(True)
        g.add_node()
        g.add_node()
        g.add_node(False, True)
        g.add_node(False, True)

        g.add_arc(0, 1, 0, 1, 1.1)
        g.add_arc(1, 2, 1, 2, 2.1)
        g.add_arc(2, 3, 2, 3, 3.1)
        g.add_arc(3, 4, 3, 4, 4.1)
        g.add_arc(4, 5, 4, gtn.epsilon, 5.1)

        g_str = [
            "0 1",
            "4 5",
            "0 1 0 1 1.1",
            "1 2 1 2 2.1",
            "2 3 2 3 3.1",
            "3 4 3 4 4.1",
            "4 5 4 -1 5.1",
        ]
        g = create_graph_from_text(g_str)
        _, tmpfile = tempfile.mkstemp()
        gtn.savetxt(tmpfile, g)
        with open(tmpfile) as f:
            lines = f.readlines()
            self.assertEqual(len(g_str), len(lines))
            for i, line in enumerate(lines):
                self.assertEqual(line.strip(), g_str[i])

    def test_loadsave(self):
        _, tmpfile = tempfile.mkstemp()

        g = gtn.Graph()
        gtn.save(tmpfile, g)
        g2 = gtn.load(tmpfile)
        self.assertTrue(gtn.equal(g, g2))

        g = gtn.Graph()
        g.add_node(True)
        g.add_node(True)
        g.add_node()
        g.add_node()
        g.add_node(False, True)
        g.add_node(False, True)

        g.add_arc(0, 1, 0, 1, 1.1)
        g.add_arc(1, 2, 1, 2, 2.1)
        g.add_arc(2, 3, 2, 3, 3.1)
        g.add_arc(3, 4, 3, 4, 4.1)
        g.add_arc(4, 5, 4, gtn.epsilon, 5.1)
        gtn.save(tmpfile, g)
        g2 = gtn.load(tmpfile)
        self.assertTrue(gtn.equal(g, g2))
        self.assertTrue(gtn.isomorphic(g, g2))


if __name__ == "__main__":
    unittest.main()
