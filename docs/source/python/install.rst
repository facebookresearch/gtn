Installing Python Bindings
--------------------------

The python package may be installed with ``pip``:

.. code-block:: shell

  pip install gtn

To build and install from source, first clone the repo:

.. code-block:: shell

  git clone https://github.com/facebookresearch/gtn.git

Setup your environment:

.. code-block:: shell

  conda create -n gtn_env
  conda activate gtn_env

Install dependencies:

.. code-block:: shell

  cd bindings/python
  conda install setuptools

Use one of the following commands for installation:

.. code-block:: shell

  python setup.py install

or, to install in editable mode (for dev):

.. code-block:: shell

  python setup.py develop

Running Python Tests
~~~~~~~~~~~~~~~~~~~~

Python binding tests can be run with ``make test``, or with

.. code-block::

    python -m unittest discover bindings/python/test


Run a simple example:

.. code-block::

   python bindings/python/examples/simple_graph.py
