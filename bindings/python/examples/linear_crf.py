#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import gtn
import numpy as np


def gen_transitions(num_classes, calc_grad=False):
    """Make a bigram transition graph."""
    g = gtn.Graph(calc_grad)
    for i in range(num_classes):
        g.add_node(False, True)
    g.add_node(True, True)
    for i in range(num_classes):
        g.add_arc(num_classes, i, i)  # s(<s>, i)
        for j in range(num_classes):
            g.add_arc(i, j, j)  # s(i, j)
    return g


def gen_potentials(num_features, num_classes, calc_grad=False):
    """Make the unary potential graph"""
    g = gtn.Graph(calc_grad)
    g.add_node(True, True)
    for i in range(num_features):
        for c in range(num_classes):
            g.add_arc(0, 0, i, c)  # f(i, c)
    return g


def gen_model(num_features, num_classes, calc_grad=False, init=True):
    transitions = gen_transitions(num_classes, calc_grad)
    potentials = gen_potentials(num_features, num_classes, calc_grad)

    # Randomly set the arc weights of the graphs:
    if init:
        transitions.set_weights(
            10 * np.random.randn(transitions.num_arcs()))
        potentials.set_weights(
            10 * np.random.randn(potentials.num_arcs()))
    return potentials, transitions


def make_chain_graph(seq, calc_grad=False):
    """Make a simple chain graph from an iterable of integers."""
    g = gtn.Graph(calc_grad)
    g.add_node(True)
    for e, s in enumerate(seq):
        g.add_node(False, e + 1 == len(seq))
        g.add_arc(e, e + 1, s)
    return g


def sample_model(
        num_features, num_classes,
        potentials, transitions,
        num_samples, max_len=20):
    """
    Sample `num_samples` from a linear-chain CRF specified
    by a `potentials` graph and a `transitions` graph. The
    samples will have a random length in `[1, max_len]`.
    """
    model = gtn.compose(potentials, transitions)

    # Draw a random X with length randomly from [1, max_len] and find the
    # most likely Y under the model:
    samples = []
    while len(samples) < num_samples:
        # Sample X:
        T = np.random.randint(1, max_len + 1)
        X = np.random.randint(0, num_features, size=(T,))
        X = make_chain_graph(X)
        # Find the most likely Y given X:
        Y = gtn.viterbi_path(gtn.compose(X, model))
        # Clean up Y:
        Y = gtn.project_output(Y)
        Y.set_weights(np.zeros(Y.num_arcs()))
        samples.append((X, Y))
    return samples


def crf_loss(X, Y, potentials, transitions):
    feature_graph = gtn.compose(X, potentials)

    # Compute the unnormalized score of `(X, Y)`
    target_graph = gtn.compose(feature_graph, gtn.intersect(Y, transitions))
    target_score = gtn.forward_score(target_graph)

    # Compute the partition function
    norm_graph = gtn.compose(feature_graph, transitions)
    norm_score = gtn.forward_score(norm_graph)

    return gtn.subtract(norm_score, target_score)


def update_params(learning_rate, *graphs):
    """Take a gradient step on each graph in `graphs`."""
    for graph in graphs:
        params = graph.weights_to_numpy()
        grad = graph.grad().weights_to_numpy()
        params += learning_rate * grad
        graph.set_weights(params)


def sampler(dataset):
    """Iterator which randomly samples from a dataset."""
    while True:
        indices = np.random.permutation(len(dataset))
        for idx in indices:
            yield dataset[idx]


def main():
    num_features = 3  # number of input features
    num_classes = 2   # number of output classes
    num_train = 1000  # size of the training set
    num_test = 200    # size of the testing set

    # Setup ground-truth model:
    gt_potentials, gt_transitions = gen_model(num_features, num_classes)

    # Sample training and test datasets:
    samples = sample_model(
        num_features, num_classes,
        gt_potentials, gt_transitions,
        num_train + num_test)
    train, test = samples[:num_train], samples[num_train:]
    print(f"Using {len(train)} samples for the training set")
    print(f"Using {len(test)} samples for the test set")

    # Make the graphs for learning:
    potentials, transitions = gen_model(
        num_features, num_classes, calc_grad=True, init=False)
    print("Unary potential graph has {} nodes and {} arcs".format(
        potentials.num_nodes(), potentials.num_arcs()))
    print("Transition graph has {} nodes and {} arcs".format(
        transitions.num_nodes(), transitions.num_arcs()))

    # Make the graphs to be learned:
    potentials, transitions = gen_model(
        num_features, num_classes, calc_grad=True, init=False)

    # Run the SGD loop:
    learning_rate = 1e-2
    max_iter = 10000
    losses = []
    for it, (X, Y) in enumerate(sampler(train)):
        # Compute the loss and take a gradient step:
        loss = crf_loss(X, Y, potentials, transitions)
        gtn.backward(loss)
        update_params(-learning_rate, potentials, transitions)

        # Clear the gradients:
        transitions.zero_grad()
        potentials.zero_grad()

        losses.append(loss.item())
        if (it + 1) % 1000 == 0:
            print("=" * 50)
            print(f"Iteration {it + 1}, Avg. Loss {np.mean(losses):.3f}")
            losses = []
        if it == max_iter:
            break

    # Evaluate on the test set:
    correct = 0.0
    total = 0
    for X, Y in test:
        full_graph = gtn.compose(gtn.compose(X, potentials), transitions)
        prediction = gtn.viterbi_path(full_graph).labels_to_list(False)
        correct += np.sum(np.array(Y.labels_to_list()) == prediction)
        total += len(prediction)
    print("Test: Accuracy {:.3f}".format(correct / total))


if __name__ == "__main__":
    main()
