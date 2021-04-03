/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

namespace gtn {
namespace detail {

namespace dataparallel {
// Exclusive/Inclusive prefix sum. The returned vector
// has one more element
int prefixSumScan(std::vector<int>& input, bool appendSum);

} // namespace dataparallel

} // namespace detail
} // namespace gtn
