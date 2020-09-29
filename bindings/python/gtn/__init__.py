#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import subprocess
import tempfile

from ._graph import *
from ._graph import __version__
from ._functions import *
from ._autograd import *
from ._utils import *
from ._rand import *
from ._creations import *
from ._parallel import *

def draw(graph, file_name, isymbols={}, osymbols={}):
    ext = os.path.splitext(file_name)[1]
    with tempfile.NamedTemporaryFile() as tmpf:
        write_dot(graph, tmpf.name, isymbols, osymbols)
        subprocess.check_call(["dot", "-T" + ext[1:], tmpf.name, "-o", file_name])
