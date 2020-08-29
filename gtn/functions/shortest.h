#pragma once

#include "gtn/graph.h"

namespace gtn {
namespace detail {

Graph shortestDistance(const Graph& g, bool tropical = false);
Graph shortestPath(const Graph& g);

} // namespace detail
} // namespace gtn
