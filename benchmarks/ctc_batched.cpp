#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "benchmarks/time_utils.h"
#include "gtn/gtn.h"

using namespace gtn;

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

std::vector<int> many_rands(int num) {
  std::vector<int> out(num);
  for (int i = 0; i < num; ++i) {
    out[i] = std::rand();
  }
  return out;
}

// For emissions generation
std::vector<float> emissions(int num) {
  std::vector<float> out(num);
  auto manyRands = many_rands(num);
  for (int i = 0; i < num; ++i) {
    auto uni = manyRands[i] / static_cast<float>(RAND_MAX);
    out[i] = uni * 10 - 5;
  }
  return out;
}

std::vector<int> rand_target(int U, int N) {
  std::vector<int> target;
  auto rands = many_rands(U);
  for (int u = 0; u < U; u++) {
    // Random integer between [1, N-1]
    auto t = rands[u] % (N - 1) + 1;
    target.push_back(t);
  }
  return target;
}

Graph ctc_graph(int U, int N, std::vector<int> target) {
  int blank = 0;
  int L = 2 * U + 1;
  Graph ctc;
  for (int l = 0; l < L; l++) {
    int idx = (l - 1) / 2;
    ctc.addNode(l == 0, l == L - 1 || l == L - 2);
    int label = l % 2 ? target[idx] : blank;
    ctc.addArc(l, l, label);
    if (l > 0) {
      ctc.addArc(l - 1, l, label);
    }
    if (l % 2 && l > 1 && label != target[idx - 1]) {
      ctc.addArc(l - 2, l, label);
    }
  }
  return ctc;
}

Graph emission_graph(int T, int N, std::vector<float> scores) {
  assert(scores.size() == T * N);
  Graph emissions;
  emissions.addNode(true);
  for (int t = 1; t <= T; t++) {
    emissions.addNode(false, t == T);
    for (int i = 0; i < N; i++) {
      emissions.addArc(t - 1, t, i, i, scores[t * N + i]);
    }
  }
  return emissions;
}

int main() {
  /* Various CTC benchmarks. */

  const int T = 1000; // input frames
  const int U = 100; // output tokens
  const int N = 28; // size of alphabet
  const int B = 8;

  // Pre-compute rand targets to avoid contention with std::rand.
  // Pre-compute emissions scores
  std::vector<std::vector<int>> targets;
  std::vector<std::vector<float>> emissionsScores;
  for (int64_t b = 0; b < B; ++b) {
    targets.push_back(rand_target(U, N));
    emissionsScores.push_back(emissions(T * N));
  }

  auto fwd = [T, U, N, &targets, &emissionsScores](
                 int b, std::vector<Graph>& vec) {
    auto ctc = ctc_graph(U, N, targets[b]);
    auto emissions = emission_graph(T, N, emissionsScores[b]);
    vec[b] = subtract(
        forwardScore(emissions), forwardScore(compose(ctc, emissions)));
  };

  auto bwd = [](int b, std::vector<Graph>& vec) {
    backward(vec[b]);
    vec[b] = Graph{}; // parallelize destruction
  };

  auto ctc_batched = [T, U, N, B, &targets, &emissionsScores, fwd, bwd]() {
    // Loss graphs
    std::vector<Graph> vec(B);
    {
      ThreadPool threadPool(B);
      for (int64_t b = 0; b < B; ++b) {
        threadPool.enqueue(fwd, b, vec);
      }
    }

    {
      ThreadPool threadPool(B);
      for (int64_t b = 0; b < B; ++b) {
        threadPool.enqueue(bwd, b, vec);
      }
    }
  };

  TIME(ctc_batched);
}
