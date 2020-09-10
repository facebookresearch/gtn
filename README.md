# gtn

`gtn` is a framework for automatic differentiation with weighted finite-state transducers (WFSTs). The framework is written in C++ and has bindings to Python.

## Getting Started

### Requirements

- A C++ compiler with good C++11 support (e.g. g++ >= 5)
- `cmake` >= 3.5.1, and `make`

### Building

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


## Python Bindings
Setting up your environment:
```
conda create -n gtn_env
conda activate gtn_env
```

Required dependencies:
```
cd bindings/python
conda install -c nogil setuptools packaging numpy
```

Use one of the following commands for installation:

```
python setup.py install
```

or, to install in editable mode (for dev):

```
python setup.py develop
```

or, to build a source distribution:

```
python setup.py sdist
```

#### Running Tests and Examples
Python binding tests can be run with `make test`, or with
```
python -m unittest discover bindings/python/test
```

Run a simple example:
```
python bindings/python/examples/simple_graph.py
```
