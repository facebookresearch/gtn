Interfacing with PyTorch
========================

Adding a GTN function or layer to PyTorch is just
like adding any
`custom extension to PyTorch <https://pytorch.org/docs/stable/notes/extending.html>`_.
For the details, take a look at an
`example <https://github.com/facebookresearch/gtn/blob/master/bindings/python/examples/pytorch_loss.py>`_
which constructs a custom loss function in PyTorch with GTN. We'll go over the
example here at a high-level with attention to the bits specific to GTN.

First declare the class which should inherit from :py:class:`torch.autograd.Function`.

.. code-block:: Python

  class GTNLossFunction(torch.autograd.Function):
      """
      A minimal example of adding a custom loss function built with GTN graphs to
      PyTorch.

      The example is a sequence criterion which computes a loss between a
      frame-level input and a token-level target. The tokens in the target can
      align to one or more frames in the input.
      """

The ``GTNLossFunction`` requires static ``forward`` and a ``backward`` methods.
The forward method, with some additional comments, is copied below:

.. code-block:: Python

    @staticmethod
    def forward(ctx, inputs, targets):
        B, T, C = inputs.shape
        losses = [None] * B
        emissions_graphs = [None] * B

        # Move data to the host as GTN operations run on the CPU:
        device = inputs.device
        inputs = inputs.cpu()
        targets = targets.cpu()

        # Compute the loss for the b-th example:
        def forward_single(b):
            emissions = gtn.linear_graph(T, C, inputs.requires_grad)
            # *NB* A reference to the `data` should be held explicitly when
            # using `data_ptr()` otherwise the memory may be claimed before the
            # weights are set. For example, the following is undefined and will
            # likely cause serious issues:
            #   `emissions.set_weights(inputs[b].contiguous().data_ptr())`
            data = inputs[b].contiguous()
            emissions.set_weights(data.data_ptr())

            target = GTNLossFunction.make_target_graph(targets[b])

            # Score the target:
            target_score = gtn.forward_score(gtn.intersect(target, emissions))

            # Normalization term:
            norm = gtn.forward_score(emissions)

            # Compute the loss:
            loss = gtn.subtract(norm, target_score)

            # We need the save the `loss` graph to call `gtn.backward` and we
            # need the `emissions` graph to access the gradients:
            losses[b] = loss
            emissions_graphs[b] = emissions

        # Compute the loss in parallel over the batch:
        gtn.parallel_for(forward_single, range(B))

        # Save some graphs and other data for backward:
        ctx.auxiliary_data = (losses, emissions_graphs, inputs.shape)

        # Put losses back in a torch tensor and move them  back to the device:
        return torch.tensor([l.item() for l in losses]).to(device)


To perform the backward computation, we save ``losses``, a list which holds the
individual graphs storing the loss for each example. We also save the
``emissions_graphs`` so that we can access the gradients in order to construct
the full gradient with respect to the ``input`` tensor.

GTN provides :py:func:`parallel_for` to run computations in parallel. We us it
here to parallelize the forward computation and backward computations over
examples in the batch. Using :py:func:`parallel_for` requires some care.  For
example notice the lines:

.. code-block:: Python

    losses = [None] * B
    emissions_graphs = [None] * B

These lists must be preconstructed so that threads can insert into the list
rather than constructively appending to it, which could cause a race condition
during the execution of ``forward_single``.

The ``backward`` method is very simple. It just calls :py:func:`backward` on
each ``loss`` graph and accumulating the gradients into a
:py:class:`torch.Tensor`.

.. code-block:: Python

    @staticmethod
    def backward(ctx, grad_output):
        losses, emissions_graphs, in_shape = ctx.auxiliary_data
        B, T, C = in_shape
        input_grad = torch.empty((B, T, C))

        # Compute the gradients for each example:
        def backward_single(b):
            gtn.backward(losses[b])
            emissions = emissions_graphs[b]
            grad = emissions.grad().weights_to_numpy()
            input_grad[b] = torch.from_numpy(grad).view(1, T, C)

        # Compute gradients in parallel over the batch:
        gtn.parallel_for(backward_single, range(B))

        return input_grad.to(grad_output.device), None

