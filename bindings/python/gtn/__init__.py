#!/usr/bin/env python3

import os
import subprocess
import tempfile

from ._graph import *
from ._graph import __version__
from ._functions import *
from ._autograd import *
from ._utils import *


def draw(graph, file_name, isymbols={}, osymbols={}):
    ext = os.path.splitext(file_name)[1]
    with tempfile.NamedTemporaryFile() as tmpf:
        write_dot(graph, tmpf.name, isymbols, osymbols)
        subprocess.check_call(["dot", "-T" + ext[1:], tmpf.name, "-o", file_name])
