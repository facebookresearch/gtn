/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <thread>

#include "gtn/parallel/parallel_map.h"

namespace gtn {
namespace detail {

const unsigned int kNumMaxThreadsDefault = 4;

size_t getNumViableThreads(size_t parallelismSize) {
  // If the query returns zero/is unsupported, fall back to a default
  unsigned int hardwareThreads = std::thread::hardware_concurrency();

  auto maxThreads =
      (hardwareThreads == 0 ? kNumMaxThreadsDefault : hardwareThreads);

  return std::min(parallelismSize, static_cast<size_t>(maxThreads));
}

ThreadPoolSingleton& ThreadPoolSingleton::getInstance() {
  static auto instance =
      ThreadPoolSingleton(getNumViableThreads(kNumMaxThreadsDefault));
  return instance;
}

void ThreadPoolSingleton::setPoolSize(size_t size) {
  if (size > size_) {
    // Create a new thread pool of the given size which implicitly waits until
    // all running work is complete
    size_ = size;
    pool_ = std::make_unique<ThreadPool>(size_);
  }
  // if we already have more threads than requested, do nothing
}

ThreadPool& ThreadPoolSingleton::get() {
  return *pool_;
}

} // namespace detail
} // namespace gtn
