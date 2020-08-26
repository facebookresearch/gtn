gtn
===

.. py:function:: add(lhs, rhs)

  Add two scalar graphs.

.. py:function:: backward(graph, grad, retain_graph=False)

  Compute the gradients of any inputs w.r.t ``grap``.

  :param graph gtn.Graph: The graph to compute gradients with respect to.
  :param grad gtn.Graph: A seed gradient, typically set to be a gradient of
    another function with respect to ``graph``.
  :param bool retain_graph: Whether or not to save the autograd graph. Setting
    this to False is more memory efficient as temporary Graphs created during
    the forward computation may be destroyed.

.. py:function:: backward(graph, retain_graph=False)
  :noindex:

  Same as :func:`backward` but with the initial ``grad`` set to be ones.

.. py:function:: draw(graph, file_name, isymbols={}, osymbols={})

.. py:function:: closure(graph)

   Compute the (Kleene) closure of the graph. This operation is recorded in the
   autograd tape.

.. py:function:: compose(first, second)

   Compose two transducers. This operation is recorded in the autograd tape.
   If ``x:y`` is transduced by ``first`` and ``y:z`` is transduced by
   ``second`` then the composed graph will transduce ``x:z``. The arc scores
   are added in the composed graph.

   Both :func:`compose` and :func:`intersect` can be much faster when operating
   on graphs with sorted arcs. See :meth:`Graph.arc_sort`.

.. py:function:: concat(graphs)

   Concatenate a list of graphs. This operation is recorded
   in the autograd tape.

   If ``x_i`` is a sequence accepted (or ``x_i:y_i`` is transduced) by ``graphs[i]``
   then the concatenated graph accepts the sequence ``x_1x_2...x_n`` if ``graphs``
   contains ``n`` graphs. The score of the path ``x_1...x_n`` is the sum of the
   scores of the individual ``x_i`` in ``graphs[i]``. The concatenated graph is
   constructuted by connecting every accepting state of ``graphs[i-1]`` to every
   starting state of ``graphs[i]`` with an epsilon transition. The starting state
   of the concatenated graphs are starting states of ``graphs[0]`` and the
   accepting states are accepting states of ``graphs[-1]``.

   Note the concatenation of 0 graphs ``gtn::concat([])`` is the graph which accepts
   the empty string (epsilon). The concatentation of a single graph is
   equivalent to a clone.

.. py:function:: concat(lhs, rhs)
  :noindex:

  Equivalent to ``concat([lhs, rhs])``, see :func:`concat`.

.. py:function:: equal(first, second)

.. py:function:: forward_score(graph)

   Compute the forward score of a graph. Returns the score in a scalar graph
   which can be accessed with :meth:`Graph.item()`. This operation is recorded
   in the autograd tape.

   The forward score is equivalent to the shortest distance from the start
   nodes to the accept nodes in the log semiring.

   **NB:** ``graph`` must be acyclic.

.. py:function:: intersect(first, second)

   Intersect two acceptors. This operation is recorded in the autograd tape.
   This function only works on acceptors, calling it on a ``graph`` where
   ``graph.ilabel(a) != graph.olabel(a)`` for some ``a`` is undefined and may yield
   incorrect results. The intersected graph accepts any path ``x`` which is
   accepted by both ``first`` and ``second``. The arc scores are added in the
   intersected graph.

   The result of :func:`compose` will yield an equivalent result, however; this
   function should be preferred since the implementation may be faster.

   Both :func:`compose` and :func:`intersect` can be much faster when operating
   on graphs with sorted arcs. See :meth:`Graph.arc_sort`.

.. py:function:: isomorphic(first, second)

.. py:function:: scalar_graph(val, calc_grad)

.. py:function:: linear_graph(M, N, calc_grad)

.. py:function:: load(file_name)

.. py:function:: negate(input)

   Negate a scalar graph.

.. py:function:: project_input(other)

   Removes the input labels from the graph and records the operation in the
   autograd tape. This function makes a copy of the input graph.

.. py:function:: project_output(other)

   Removes the output labels from the graph and records the operation in the
   autograd tape. This function makes a copy of the input graph.

.. py:function:: remove(other, label=gtn.epsilon)

   Construct the equivalent graph without epsilon transitions. The epsilon
   closure of each node in the graph is computed and the required transitions
   are added to yield the epsilon-free equivalen graph. If ``label`` is
   specified then instead of removing epsilon transitions, arcs with the
   matching label are removed. The removed arc labels are treated as if they
   were epsilon transitions.

.. py:function:: subtract(lhs, rhs)

   Subtract one scalar graph from another.

.. py:function:: union(graphs)

   Construct the union of a list of graphs.

.. py:function:: viterbi_score(graph)

   Compute the Viterbi score of a graph. Returns the score in a scalar graph
   which can be accessed with :meth:`Graph.item()`. This operation is recorded
   in the autograd tape.

   This is equivalent to the shortest distance from the start nodes to the
   accepting nodes in the tropical semiring.

   **NB:** ``graph`` must be acyclic.

.. py:function:: viterbi_path(graph)

   Compue the Viterbi shortest path of a graph and return it in a single chain
   graph with the labels and weights of the shortest path. This operation is
   recorded in the autograd tape.

   The Viterbi shorted path is equivalent to the shortest path from the start
   nodes to the accepting nodes in the tropical semiring.

   **NB:** ``graph`` must be acyclic.

.. py:function:: write_dot(graph, file_name, isymbbols={}, osymbols={})
