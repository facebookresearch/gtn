#!/usr/bin/env python3

import unittest
import sys


class GTNModuleTestCase(unittest.TestCase):
    def setUp(self):
        """
        If importing the GTN module failed, skip the test
        """
        if "gtn" not in sys.modules:
            raise unittest.SkipTest("GTN module not imported - skipping test")

    def assertListAlmostEqual(self, list1, list2, places=7):
        self.assertEqual(len(list1), len(list2))
        for i in range(0, len(list2)):
            self.assertAlmostEqual(list1[i], list2[i], places=4)
