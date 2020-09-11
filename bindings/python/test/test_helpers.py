#!/usr/bin/env python3

import tempfile
import gtn


def create_graph_from_text(txt_arr):
    graph = None
    with tempfile.NamedTemporaryFile(mode="w") as fid:
        for l in txt_arr:
            fid.write(l + "\n")
        fid.flush()
        graph = gtn.loadtxt(fid.name)
    assert graph
    return graph
