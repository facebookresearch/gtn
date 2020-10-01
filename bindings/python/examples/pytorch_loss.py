import gtn

try:
    import torch
except ImportError:
    print("You must install PyTorch to run this example.")
    import sys
    sys.exit(0)


class GTNLossFunction(torch.autograd.Function):
    """
    A minimal example of adding a custom loss function built with GTN graphs to
    PyTorch.

    The example is a sequence criterion which computes a loss between a
    frame-level input and a token-level target. The tokens in the target can
    align to one or more frames in the input.
    """

    @staticmethod
    def make_target_graph(target):
        """
        Construct the target graph for the sequence in target. Each token in
        target can align to one or more input frames.
        """
        g = gtn.Graph(False)
        L = len(target)
        g.add_node(True)
        for l in range(1, L + 1):
            g.add_node(False, l == L)
            g.add_arc(l - 1, l, target[l - 1])
            g.add_arc(l, l, target[l - 1])
        g.arc_sort(True)
        return g

    @staticmethod
    def forward(ctx, inputs, targets):
        B, T, C = inputs.shape
        losses = [None] * B
        emissions_graphs = [None] * B

        # Move data to the host:
        device = inputs.device
        inputs = inputs.cpu()
        targets = targets.cpu()

        # Compute the loss for the b-th example:
        def forward_single(b):
            emissions = gtn.linear_graph(T, C, inputs.requires_grad)
            data = inputs[b].contiguous()
            emissions.set_weights(data.data_ptr())

            target = GTNLossFunction.make_target_graph(targets[b])

            # Score the target:
            target_score = gtn.forward_score(gtn.intersect(target, emissions))

            # Normalization term:
            norm = gtn.forward_score(emissions)

            # Compute the loss:
            loss = gtn.subtract(norm, target_score)

            # Save state for backward:
            losses[b] = loss
            emissions_graphs[b] = emissions

        # Compute the loss in parallel over the batch:
        gtn.parallel_for(forward_single, range(B))

        ctx.auxiliary_data = (losses, emissions_graphs, inputs.shape)

        # Put losses back in a torch tensor and move them  back to the device:
        return torch.tensor([l.item() for l in losses]).to(device)


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


# make an alias for the loss function:
GTNLoss = GTNLossFunction.apply


def main():
    B = 4  # batch size
    T = 10  # input sequence length
    U = 5  # target sequence length
    C = 20  # number of output tokens
    inputs = torch.randn(B, T, C, requires_grad=True)
    target = torch.randint(high=C, size=(B, U), )

    loss = GTNLoss(inputs, target).mean()
    loss.backward()

    print(f"Loss {loss.item():.3f}")
    print(f"Grad has shape {tuple(inputs.grad.shape)}")


if __name__ == "__main__":
    main()
