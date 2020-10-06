gtn
===

Autograd
--------

.. py:function:: backward(g, grad, retain_graph=False)

  Compute the gradients of any inputs with respect to ``graph``.

  :param Graph graph: The graph to compute gradients with respect to.
  :param Graph grad: A seed gradient, typically set to be a gradient of
    another function with respect to ``graph``.
  :param bool retain_graph: Whether or not to save the autograd graph. Setting
    this to False is more memory efficient as temporary graphs created during
    the forward computation may be destroyed.

.. py:function:: backward(g, retain_graph=False)
  :noindex:

  Same as :func:`backward` but with the initial ``grad`` set to be ones.


Functions
---------

.. py:function:: add(g1, g2)

  Add two scalar graphs.

.. py:function:: closure(g)

   Compute the (Kleene) closure of the graph. This operation is recorded in the
   autograd tape.

.. py:function:: compose(g1, g2)

   Compose two transducers. This operation is recorded in the autograd tape.
   If ``x:y`` is transduced by ``g1`` and ``y:z`` is transduced by
   ``g2`` then the composed graph will transduce ``x:z``. The arc scores
   are added in the composed graph.

   Both :func:`compose` and :func:`intersect` can be much faster when operating
   on graphs with sorted arcs. See :meth:`Graph.arc_sort`.

.. py:function:: concat(graphs)

   Concatenate a list of graphs. This operation is recorded
   in the autograd tape.

   If ``x_i`` is a sequence accepted (or ``x_i:y_i`` is transduced) by
   ``graphs[i]`` then the concatenated graph accepts the sequence
   ``x_1x_2...x_n`` if ``graphs`` contains ``n`` graphs. The score of the path
   ``x_1...x_n`` is the sum of the scores of the individual ``x_i`` in
   ``graphs[i]``. The concatenated graph is constructed by connecting every
   accepting state of ``graphs[i-1]`` to every starting state of ``graphs[i]``
   with an :math:`\epsilon` transition. The starting state of the concatenated
   graphs are starting states of ``graphs[0]`` and the accepting states are
   accepting states of ``graphs[-1]``.

   Note the concatenation of 0 graphs ``gtn::concat([])`` is the graph which accepts
   the empty string (:math:`\epsilon`). The concatenation of a single graph is
   equivalent to a clone.

.. py:function:: concat(g1, g2)
  :noindex:

  Equivalent to ``concat([g1, g2])``, see :func:`concat`.

.. py:function:: forward_score(g)

   Compute the forward score of a graph. Returns the score in a scalar graph
   which can be accessed with :meth:`Graph.item()`. This operation is recorded
   in the autograd tape.

   The forward score is equivalent to the shortest distance from the start
   nodes to the accept nodes in the log semiring.

   **NB:** ``graph`` must be acyclic.

.. py:function:: intersect(g1, g2)

   Intersect two acceptors. This operation is recorded in the autograd tape.
   This function only works on acceptors, calling it on a ``graph`` where
   ``graph.ilabel(a) != graph.olabel(a)`` for some ``a`` is undefined and may yield
   incorrect results. The intersected graph accepts any path ``x`` which is
   accepted by both ``g1`` and ``g2``. The arc scores are added in the
   intersected graph.

   The result of :func:`compose` will yield an equivalent result, however; this
   function should be preferred since the implementation may be faster.

   Both :func:`compose` and :func:`intersect` can be much faster when operating
   on graphs with sorted arcs. See :meth:`Graph.arc_sort`.

.. py:function:: negate(g)

   Negate a scalar graph.

.. py:function:: project_input(other)

   Removes the input labels from the graph and records the operation in the
   autograd tape. This function makes a copy of the input graph.

.. py:function:: project_output(other)

   Removes the output labels from the graph and records the operation in the
   autograd tape. This function makes a copy of the input graph.

.. py:function:: remove(other, label=gtn.epsilon)

   Construct the equivalent graph without :math:`\epsilon` transitions. The
   :math:`\epsilon` closure of each node in the graph is computed and the
   required transitions are added to yield the :math:`\epsilon`-free equivalent
   graph. If ``label`` is specified then instead of removing epsilon
   transitions, arcs with the matching label are removed. The removed arc
   labels are treated as if they were :math:`\epsilon` transitions.

.. py:function:: subtract(g1, g2)

   Subtract one scalar graph from another.

.. py:function:: union(graphs)

   Construct the union of a list of graphs.

.. py:function:: viterbi_score(g)

   Compute the Viterbi score of a graph. Returns the score in a scalar graph
   which can be accessed with :meth:`Graph.item()`. This operation is recorded
   in the autograd tape.

   This is equivalent to the shortest distance from the start nodes to the
   accepting nodes in the tropical semiring.

   **NB:** ``graph`` must be acyclic.

.. py:function:: viterbi_path(g)

   Compue the Viterbi shortest path of a graph and return it in a single chain
   graph with the labels and weights of the shortest path. This operation is
   recorded in the autograd tape.

   The Viterbi shorted path is equivalent to the shortest path from the start
   nodes to the accepting nodes in the tropical semiring.

   **NB:** ``graph`` must be acyclic.


Creations
---------

.. py:function:: scalar_graph(weight, calc_grad = True)

  Creates a scalar graph - a graph with a single arc between two nodes with a
  given weight value and an :math:`\epsilon` label (:data:`epsilon`).

.. py:function:: linear_graph(M, N, calc_grad = True)

  Create a linear chain graph with ``M + 1`` nodes and ``N`` edges between each
  node.  The labels of the edges between each node are the integers ``[0, ...,
  N - 1]``.


Comparisons
-----------

.. py:function:: equal(g1, g2)

  Checks if two graphs are exactly equal (not isomorphic).

.. py:function:: isomorphic(g1, g2)

  Checks if two graphs are isomorphic. This function will be extremely slow for
  large graphs.


Parallel	
--------	

.. py:function:: parallel_for(function, int_list)

   Computes the result of a given function that takes an int argument in
   parallel given some list of ints over which to process.

   Returns nothing, even if the passed function has a return value.


Input and Output
----------------

.. py:function:: draw(g1, file_name, isymbols={}, osymbols={})

  Draw a graph to an image. This function requires a working installation of
  `Graphviz <https://graphviz.org/>`_. Arc labels are of the format
  ``ilabel/olabel:weight``. If the output symbols are not specified then the
  ``olabel`` is omitted and arc labels are of the format ``ilabel:weight``. If
  the input symbols are not specified then integer ids are used as the label.

  The format of the image is determined by the `file_name` extension and can be
  any dot supported extension (check with ``dot -T?``).

  :param Graph graph: The graph to draw
  :param str file_name: The name of the file to write to
  :param dict isymbols: A map of integer ids to strings used for arc input labels
  :param dict osymbols: A map of integer ids to strings used for arc output labels

.. py:function:: load(file_name)

  Load a graph from a file. The first two lines contain the list of space
  separated start and accept nodes respectively. The following lines contain
  the arcs in the format:
  ::

    srcNode dstNode ilabel [olabel=ilabel] [weight=0.0]

  where ``[x=y]`` indicate optional values for ``x`` with a default value of
  ``y``.

  For example:
  ::

    0
    1
    0 1 1

  is a two node graph with an arc from start node 0 to accept node 1 with
  input and output label of 1,
  ::

    0
    1
    0 1 1 2

  is a two node graph with an arc from node 0 to node 1 with input label 1
  and output label 2, and
  ::

    0
    1
    0 1 1 2 3.0

  is a two node graph with an arc from node 0 to node 1 with input label 1,
  output label 2, and a weight of 3.0.

.. py:function:: write_dot(g, file_name, isymbols={}, osymbols={})

  Write the graph in `Graphviz <https://graphviz.org/>`_ DOT format. See
  :func:`draw`.

.. py:data:: epsilon
  :type: int

  Use the :data:`epsilon` constant to refer to :math:`\epsilon`
  transitions.

