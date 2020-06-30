#pragma once

#include "graph.h"

namespace gtn {
namespace detail {

Graph shortestDistance(Graph graph, bool tropical = false);
Graph shortestPath(Graph graph);

}
}
