
#pragma once

#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtn/graph.h"
#include "gtn/parallel/thread_pool.h"

namespace gtn {

namespace detail {

size_t getNumViableThreads(size_t parallelismSize);

class ThreadPoolSingleton {
 public:
  ThreadPoolSingleton(size_t size) : size_(size), pool_(new ThreadPool(size)) {}

  /**
   * @return the ThreadPoolSingleton instance
   */
  static ThreadPoolSingleton& getInstance();

  /**
   * Sets the size of the thread pool. If the requested size is smaller than the
   * number of threads of the existing thread pool, this operation is a no-op.
   * If the requested size is larger than the current thread pool size, the
   * existing thread pool is destroyed (which synchronizes all existing work)
   * and a new thread pool with the requested size is created.
   *
   * @param[in] size the requested size of the thread pool
   */
  void setPoolSize(size_t size);

  /**
   * @return the underlying thread pool instance
   */
  ThreadPool& get();

 private:
  size_t size_;
  std::unique_ptr<ThreadPool> pool_;
};

} // namespace detail

namespace {

// Small trait to fix an issue with binding non-const lvalue references of
// fundamental types to their corresponding rvalues
template <typename T>
struct ret {
  using type =
      typename std::conditional<std::is_fundamental<T>::value, T, T&>::type;
};

template <typename T>
typename ret<T>::type
getIdxOrBroadcast(size_t size, size_t idx, std::vector<T>& in) {
  if (in.size() == size) {
    return in[idx];
  } else if (in.size() == 1) {
    return in[0];
  } else {
    throw std::runtime_error(
        "parallelMap getIdxOrBroadcast got invalid size "
        "or unbroadcastable vector");
  }
}

// Base case
template <typename T>
T&& max(T&& val) {
  return std::forward<T>(val);
}

// Max of a variable number of parameters
template <typename T0, typename T1, typename... Ts>
typename std::common_type<T0, T1, Ts...>::type
max(T0&& val1, T1&& val2, Ts&&... vs) {
  return (val1 > val2) ? max(val1, std::forward<Ts>(vs)...)
                       : max(val2, std::forward<Ts>(vs)...);
}

template <typename T>
size_t vecSize(T& in) {
  return in.size();
}

template <typename T>
struct OutPayload {
  std::vector<T> val;

  OutPayload(size_t outSize) : val(outSize) {}

  template <typename F, typename... Args>
  void compute(size_t idx, F&& f, Args&&... args) {
    val[idx] = f(std::forward<Args>(args)...);
  }

  std::vector<T> value() const {
    return val;
  }
};

// A void specialization is required to handle functions that return void
template <>
struct OutPayload<void> {
  OutPayload(size_t) {}

  template <typename F, typename... Args>
  void compute(size_t, F&& f, Args&&... args) {
    f(std::forward<Args>(args)...);
  }

  void value() const {}
};

} // namespace

template <typename FuncType, typename... Args>
auto parallelMap(FuncType&& function, Args&&... inputs) {
  // Maximum input size in number of elements
  const auto size = max(vecSize(inputs)...);

  using OutType =
      decltype(std::declval<FuncType&&>()(getIdxOrBroadcast(1, 0, inputs)...));

  OutPayload<OutType> out(size);
  std::vector<std::future<void>> futures(size);

  auto& threadPool = detail::ThreadPoolSingleton::getInstance();
  threadPool.setPoolSize(detail::getNumViableThreads(size));

  for (size_t i = 0; i < size; ++i) {
    futures[i] = threadPool.get().enqueue(
        [size, i, &out, &function](Args&&... inputs) {
          out.compute(i, function, getIdxOrBroadcast(size, i, inputs)...);
        },
        std::forward<Args>(inputs)...);
  }

  // Wait on all work to be done before returning
  for (auto& future : futures) {
    future.wait();
  }
  return out.value();
}

} // namespace gtn
