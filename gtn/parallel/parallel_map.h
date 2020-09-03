
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
/**
 * \addtogroup parallel
 * @{
 */

namespace detail {

size_t getNumViableThreads(size_t parallelismSize);

/**
 * A singleton that stores a globally-accessible thread pool for reuse.
 */
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
  using type = typename std::
      conditional<std::is_fundamental<T>::value, const T, const T&>::type;
};

template <typename T>
typename ret<T>::type
getIdxOrBroadcast(size_t size, size_t idx, const std::vector<T>& in) {
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
size_t getSize(const T& in) {
  return in.size();
}

template <typename T>
struct OutPayload {
  std::vector<T> out;

  OutPayload(std::vector<std::future<T>>& futures) {
    const size_t size = futures.size();
    out.reserve(size);
    for (size_t i = 0; i < size; ++i) {
      out.push_back(futures[i].get());
    }
  }

  std::vector<T> value() const {
    return out;
  }
};

// A void specialization is required to handle functions that return void
template <>
struct OutPayload<void> {
  OutPayload(std::vector<std::future<void>>& futures) {
    for (size_t i = 0; i < futures.size(); ++i) {
      futures[i].wait();
    }
  }

  void value() const {}
};

} // namespace

/**
 * Executes a function in parallel.
 *
 * @param[in] function A function pointer to execute in parallel
 * @param[in] ...inputs variadic arguments of iterable/indexable containers,
 * such as `std::vector`s, i.e. `vector<T1>, vector<T2>,...`. Types must match
 * the input types of function exactly, i.e. `function` must take arguments
 * ``T1, T2,...``.
 *
 * @return a vector of type `T` where `T` is the type of the output type of
 * `function`. If the given function returns `void`, the return type is `void`.
 */
template <typename FuncType, typename... Args>
auto parallelMap(FuncType&& function, Args&&... inputs) {
  // Maximum input size in number of elements
  const auto size = max(getSize(inputs)...);

  using OutType =
      decltype(std::declval<FuncType&&>()(getIdxOrBroadcast(1, 0, inputs)...));
  std::vector<std::future<OutType>> futures(size);

  auto& threadPool = detail::ThreadPoolSingleton::getInstance();
  threadPool.setPoolSize(detail::getNumViableThreads(size));

  for (size_t i = 0; i < size; ++i) {
    futures[i] = threadPool.get().enqueue(
        [size, i, &function](Args&&... inputs) -> OutType {
          return function(getIdxOrBroadcast(size, i, inputs)...);
        },
        std::forward<Args>(inputs)...);
  }

  // Waits until work is done
  return OutPayload<OutType>(futures).value();
}

/**
 * @}
 */

} // namespace gtn
