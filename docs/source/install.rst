Building and Installing
=======================

Build Requirements
------------------

- A C++ compiler with good C++14 support (e.g. g++ >= 5.0)
- `cmake <https://cmake.org/>`_ -- version 3.5.1 or later, and ``make``

Building and Installing with CMake
----------------------------------

Building
~~~~~~~~

Currently, GTN must be built and installed from source.

First, clone gtn from `its repository on Github <https://github.com/facebookresearch/gtn>`_:

.. code-block:: shell

   git clone https://github.com/facebookresearch/gtn.git && cd gtn

Create a build directory and run CMake and make:

.. code-block:: shell

   mkdir -p build && cd build
   cmake ..
   make -j $(nproc)

Run tests with:

.. code-block:: shell

   make test

Install with:

.. code-block:: shell

   make install


Build Options
~~~~~~~~~~~~~

+---------------------------+-----------------------------------------------+---------------+
| Options                   | Configurations                                | Default Value |
+===========================+===============================================+===============+
| GTN_BUILD_TESTS           | ON, OFF                                       | ON            |
+---------------------------+-----------------------------------------------+---------------+
| GTN_BUILD_BENCHMARKS      | ON, OFF                                       | ON            |
+---------------------------+-----------------------------------------------+---------------+
| GTN_BUILD_EXAMPLES        | ON, OFF                                       | ON            |
+---------------------------+-----------------------------------------------+---------------+
| GTN_BUILD_PYTHON_BINDINGS | ON, OFF                                       | OFF           |
+---------------------------+-----------------------------------------------+---------------+
| CMAKE_BUILD_TYPE          | `See CMake Docs <https://bit.ly/3gwYuk9>`_    | Debug         |
+---------------------------+-----------------------------------------------+---------------+


Linking your Project with CMake
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once flashlight is built and installed, including it in another project is simple with a CMake imported target. In your CMake list, add:

.. code-block:: cmake

   find_package(gtn REQUIRED)

   # Create myCompiledTarget, etc.
   # ...

   target_link_libraries(myCompiledTarget PUBLIC gtn::gtn)

Your target's files will be linked with the library and can include headers (e.g. ``#include <gtn/gtn.h>``) directly.


Python Bindings
---------------

:doc:`python/install`
