/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gtn/graph.h"

namespace gtn {
namespace detail {

Graph shortestDistance(const Graph& g, bool tropical = false);
Graph shortestPath(const Graph& g);

} // namespace detail
} // namespace gtn
