#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gtn/gtn.h"

using namespace gtn;

#define TIME(FUNC)                                           \
  std::cout << "Timing " << #FUNC << " ...  " << std::flush; \
  std::cout << std::setprecision(5) << timeit(FUNC) << " msec" << std::endl;

#define milliseconds(x) \
  std::chrono::duration_cast<std::chrono::milliseconds>(x).count()
#define timeNow() std::chrono::high_resolution_clock::now()

double timeit(std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 5; ++i) {
    fn();
  }

  int numIters = 100;
  auto start = timeNow();
  for (int i = 0; i < numIters; i++) {
    fn();
  }
  auto end = timeNow();
  return milliseconds(end - start) / static_cast<double>(numIters);
}

Graph makeLinear(int M, int N) {
  Graph linear;
  linear.addNode(true);
  for (int m = 1; m <= M; m++) {
    linear.addNode(false, m == M);
    for (int n = 0; n < N; n++) {
      linear.addArc(m - 1, m, n);
    }
  }
  return linear;
}

// *NB* num_arcs is assumed to be greater than num_nodes.
Graph makeRandomDAG(int num_nodes, int num_arcs) {
  Graph graph;
  graph.addNode(true);
  for (int n = 1; n < num_nodes; n++) {
    graph.addNode(false, n == num_nodes - 1);
    graph.addArc(n - 1, n, 0); // assure graph is connected
  }
  for (int i = 0; i < num_arcs - num_nodes + 1; i++) {
    // To preserve DAG property, select src then select dst to be
    // greater than source.
    // select from [0, num_nodes-2]:
    auto src = rand() % (num_nodes - 1);
    // then select from  [src + 1, num_nodes - 1]:
    auto dst = src + 1 + rand() % (num_nodes - src - 1);
    graph.addArc(src, dst, 0);
  }
  return graph;
}

namespace {

/**
 * A simple thread pool implementation from
 * https://github.com/progschj/ThreadPool for use in benchmarking
 * batch-parallelism across threads.
 */
class ThreadPool {
 public:
  ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i)
      workers.emplace_back([this] {
        for (;;) {
          std::function<void()> task;

          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(
                lock, [this] { return this->stop || !this->tasks.empty(); });
            if (this->stop && this->tasks.empty())
              return;
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }

          task();
        }
      });
  }

  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);

      // don't allow enqueueing after stopping the pool
      if (stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");

      tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
  }
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers)
      worker.join();
  }

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::queue<std::function<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

} // namespace
