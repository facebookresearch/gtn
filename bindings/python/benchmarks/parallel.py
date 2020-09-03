import gtn

from time_utils import time_func


def time_compose():
    N1 = 100
    N2 = 50
    A1 = 20
    A2 = 500
    graphs1 = [gtn.linear_graph(N1, A1) for _ in range(B)]
    graphs2 = [gtn.linear_graph(N2, A2) for _ in range(B)]
    for g in graphs2:
        for i in range(N2):
            for j in range(A2):
              g.add_arc(i, i, j);
    def fwd():
        gtn.parallel_map(gtn.compose, graphs1, graphs2)
    time_func(fwd, 20, "parallel compose Fwd")

    out = gtn.parallel_map(gtn.compose, graphs1, graphs2)

    def bwd():
        gtn.parallel_map(gtn.backward, out, [True])
    time_func(bwd, 20, "parallel compose bwd")


def time_forward_score():
    graphs = [gtn.linear_graph(1000, 100) for _ in range(B)]
    def fwd():
        gtn.parallel_map(gtn.forward_score, graphs)
    time_func(fwd, 100, "parallel forward_score Fwd")

    out = gtn.parallel_map(gtn.forward_score, graphs)

    def bwd():
        gtn.parallel_map(gtn.backward, out, [True])
    time_func(bwd, 100, "parallel forward_score bwd")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        B = int(sys.argv[1])
    else:
        B = 1
    time_compose()
    time_forward_score()
