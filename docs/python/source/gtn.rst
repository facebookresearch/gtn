gtn
===

.. py:function:: add(lhs, rhs)

  Add two scalar graphs.

  :param Graph lhs: The first input graph
  :param Graph rhs: The second input graph

.. py:function:: backward(graph, retain_graph=False)
.. py:function:: backward(graph, grad, retain_graph=False)
  :noindex:

.. py:function:: draw(graph, file_name, isymbols={}, osymbols={})

.. py:function:: closure(graph)

.. py:function:: compose(first, second)

.. py:function:: concat(lhs, rhs)

.. py:function:: concat(graphs)
  :noindex:

.. py:function:: equal(first, second)

.. py:function:: forward_score(graph)

.. py:function:: intersect(first, second)

.. py:function:: isomorphic(first, second)

.. py:function:: linear_graph(M, N, calc_grad)

.. py:function:: load(file_name)

.. py:function:: negate(input)

.. py:function:: project_input(other)

.. py:function:: project_output(other)

.. py:function:: remove(other, label=gtn.epsilon)

.. py:function:: subtract(lhs, rhs)

.. py:function:: sum(graphs)

.. py:function:: viterbi_score(graph)

.. py:function:: viterbi_path(graph)

.. py:function:: write_dot(graph, file_name, isymbbols={}, osymbols={})
