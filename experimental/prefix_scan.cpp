/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>

namespace gtn {
namespace detail {
namespace dataparallel {

// Exclusive/Inclusive prefix sum. The returned vector
// has one more element
int prefixSumScan(std::vector<int>& input, bool appendSum) {
  int sum = 0;
  for (size_t i = 0; i < input.size(); ++i) {
    auto count = input[i];
    input[i] = sum;
    sum += count;
  }
  if (appendSum) {
    input.push_back(sum);
  }

  return sum;
}

} // namespace dataparallel
} // namespace detail
} // namespace gtn
