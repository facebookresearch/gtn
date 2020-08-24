gtn.Graph
=========

.. py:class:: Graph(calc_grad=True)

  A weighted finite state acceptor or transducer.

  :param bool calc_grad: Specify if a gradient is required for the graph.

  .. py:method:: add_node(start=False, accept=False)

    Add a node to the graph.

    :param int start: Mark the node as a starting state
    :param int accept: Mark the node as an accepting state
    :return: The integer id of the node.
    :rtype: int

  .. py:method:: add_arc(up_node, down_node, label)
    :noindex:

    Add an accepting arc to the graph.

    :param int up_node: The id of the source node
    :param int down_nodw: The id of the destination node
    :param int label: Label for the arc
    :return: The integer id of the arc.

  .. py:method:: add_arc(up_node, down_node, ilabel, olabel, weight=0.0)

    Add a transducing arc to the graph.

    :param int up_node: The id of the source node
    :param int down_nodw: The id of the destination node
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

  .. py:method:: item()

    Get the weight on a single arc graph.
