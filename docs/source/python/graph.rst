gtn.Graph
=========

.. py:class:: Graph(calc_grad=True)

  A weighted finite-state acceptor or transducer.

  :param bool calc_grad: Specify if a gradient is required for the graph.


  .. py:method:: Graph.add_node(start=False, accept=False)

    Add a node to the graph.

    :param int start: Mark the node as a starting state
    :param int accept: Mark the node as an accepting state
    :return: The integer id of the node.
    :rtype: int

  .. py:method:: add_arc(src_node, dst_node, label)
    :noindex:

    Add an accepting arc to the graph.

    :param int src_node: The id of the source node
    :param int dst_node: The id of the destination node
    :param int label: Label for the arc
    :return: The integer id of the arc.

  .. py:method:: add_arc(src_node, dst_node, ilabel, olabel, weight=0.0)

    Add a transducing arc to the graph.

    :param int src_node: The id of the source node
    :param int dst_node: The id of the destination node
    :param int ilabel: Input label for the arc
    :param int olabel: Output label for the arc
    :param float weight: Weight for the arc
    :return: The integer id of the arc.

  .. py:method:: arc_sort(olabel=False)

    Sort the arcs entering and exiting a node by label.

    This function is intended to be used prior to calls to :func:`intersect`
    and :func:`compose` to improve the efficiency of the algorithms.

    :param bool olabel: Sort by increasing order on the arc input label
      (default) or output label if ``olabel == True``.

  .. py:method:: mark_arc_sorted(olabel=False)

    Mark the arcs entering and exiting the nodes of a graph as sorted. This
    method is intended to be used when a graph is constructed in sorted order
    to avoid paying for a call to :meth:`arc_sort`.

    :param bool olabel: Mark as sorted by input label (default) or output
      label if ``olabel == True``.

  .. py:method:: item()

    Get the weight on a single arc graph.

  .. py:method:: num_arcs()

    Get the number of arcs in the graph.

  .. py:method:: num_nodes()

    Get the number of nodes in the graph.

  .. py:method:: num_start()

    Get the number of starting nodes in the graph.

  .. py:method:: num_accept()

    Get the number of accepting nodes in the graph.

  .. py:method:: weights()

    Get a pointer to an array of the graph's arc weights.

  .. py:method:: weights_to_list()

    Get a :class:`list` of the graph's arc weights.

  .. py:method:: weights_to_numpy()

    Get a 1D :class:`numpy.ndarray` with the graph's arc weights.

  .. py:method:: set_weights(weights)

    Set all of the arc weights of the graph.

    :param weights: The weights of the arcs to set. An :class:`int` type is
      treated as the pointer to the first entry of an array of weights.
    :type weights: int or list or numpy.ndarray

  .. py:method:: labels_to_list(ilabel = True)

    Get the graph's arc labels as a list.

    :param bool ilabel: If `True` return the input labels,
      otherwise return the output labels.

  .. py:attribute:: calc_grad
    :type: bool

    Set to ``True`` to compute the gradient for the graph.


  .. py:method:: grad()

    Access the graph's gradient :class:`Graph`.

  .. py:method:: zero_grad()

    Clear the graph's gradient.
