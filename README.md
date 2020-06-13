# gtn

## Building

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


## Using python bindings:
```
git submodule init && git submodule update
cd bindings/python
pip install -e .

#### Running examples
python /private/home/vineelkpratap/gtn/gtn/bindings/python/examples/simple_graph.py
```
