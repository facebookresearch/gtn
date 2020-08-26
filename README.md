# gtn

`gtn` is a framework with which to build differentiable weighted finite-state transducers (WFSTs). The framework is written in C++ and has bindings to Python.

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
conda create -n gtn_env -c nogil python=3.9
conda activate gtn_env
```
Required dependencies:
```
cd bindings/python
pip install packaging numpy
```

Install bindings:
```
pip install -e .
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

### Building Python Bindings from Scratch
To develop on Python bindings, follow the steps above to create your environment and install dependencies, then run:
```
python setup.py install
pip install -e .
```

### Building the Documentation
#### Environment and Dependencies
Setup a Python environment:
```
conda create -n docs
conda activate docs
```
Install dependencies:
```
pip install sphinx
pip install breathe
pip install sphinx_rtd_theme
```

#### Building
Depending on which docs you'd like to build, navigate to the `docs/cpp` or `docs/python` directory in the project root. Then run:
```
doxygen && make html -j$(nproc)
```

Navigate into the resulting `build/html` directory, then run:
```
python -m http.server <port>
```

The generated docs can be viewed at `localhost:<port>` in your browser.