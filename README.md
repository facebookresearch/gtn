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


## Using Python bindings:
### Build Bindings
```
cmake .. \
      -DGTN_BUILD_PYTHON_BINDINGS=ON \
      -DGTN_BUILD_PYTHON_BINDINGS_NO_GIL=ON # optional, for best performance
```

### Install Bindings
```
cd bindings/python
pip install -e .

#### Running examples
python /private/home/vineelkpratap/gtn/gtn/bindings/python/examples/simple_graph.py
```
