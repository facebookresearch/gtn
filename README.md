<div align="center">
<img src="gtn.svg" alt="logo" width="300"></img>
</div>

# GTN: Automatic Differentiation with WFSTs

[**Quickstart**](#quickstart)
| [**Installation**](#installation)
| [**Documentation**](https://gtn.readthedocs.io/en/latest/)

## What is GTN?

GTN is a framework for automatic differentiation with weighted finite-state
transducers. The framework is written in C++ and has bindings to
Python.

The goal of GTN is to make adding and experimenting with structure in learning
algorithms much simpler. This structure is encoded as weighted automata, either
acceptors (WFSAs) or transducers (WFSTs). With `gtn` you can dynamically construct complex
graphs from operations on simpler graphs. Automatic differentiaation gives gradients with repsect to any input or intermediate graph
with a single call to `gtn.backward`.

## Quick Start

First [install](#installation) the python bindings.

The following is a minimal example of building two WFSAs with `gtn`, constructing a simple function on the graphs, and computing gradients.

```python
import gtn

# Make some graphs:
g1 = gtn.Graph()
g1.add_node(True)  # Add a start node
g1.add_n_ode()  # Add an internal node
g1.add_node(False, True)  # Add an accepting node

# Add arcs with (src node, dst node, label):
g1.add_arc(0, 1, 1)
g1.add_arc(0, 1, 2)
g1.add_arc(1, 2, 1)
g1.add_arc(1, 2, 0)

g2 = gtn.Graph()
g2.add_node(True)
g2.add_arc(0, 0, 1)
g2.add_arc(0, 0, 0)

# Compute a function of the graphs:
intersection = gtn.intersect(g1, g2)
score = gtn.forward_score()

# Visualize the intersected graph:
gtn.draw(intersection, "intersection.pdf")

# Backprop:
gtn.backward(score)
```


## Installation

### Requirements

- A C++ compiler with good C++11 support (e.g. g++ >= 5)
- `cmake` >= 3.5.1, and `make`

### Python

Install the Python bindings with

```
pip install gtn
```

### Building C++ from source

First, clone the project:

```
git clone git@github.com:facebookresearch/gtn.git
cd gtn
```

Create a build dir:

```
mkdir build && cd build
```

Run CMake (no dependencies):

```
cmake ..
```

Build:

```
make -j8
```

Run tests:

```
make test
```

### Python bindings from source

Setting up your environment:
```
conda create -n gtn_env
conda activate gtn_env
```

Required dependencies:
```
cd bindings/python
conda install setuptools
```

Use one of the following commands for installation:

```
python setup.py install
```

or, to install in editable mode (for dev):

```
python setup.py develop
```

Python binding tests can be run with `make test`, or with
```
python -m unittest discover bindings/python/test
```

Run a simple example:
```
python bindings/python/examples/simple_graph.py
```
