# gtn

## Getting Started

### Requirements

- A C++ compiler with good C++11 support (e.g. g++ >= 5)
- `cmake` >= 3.5.1, and `make`

### Building

First, clone the project:

```
git clone git@github.com:awni/gtn.git
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
Install:
```
cd bindings/python
pip install packaging
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
