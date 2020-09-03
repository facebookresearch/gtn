import time

def time_func(func, iterations=100, name=None):
    # warmup:
    for i in range(5):
        func()

    start = time.perf_counter()
    for i in range(iterations):
        func()
    time_taken = (time.perf_counter() - start) * 1e3 / iterations
    name = "function" if name is None else name
    print("\"{}\" took {:.3f} (ms)".format(name, time_taken))
