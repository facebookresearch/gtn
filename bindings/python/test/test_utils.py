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
