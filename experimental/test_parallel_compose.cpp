#include <cstdlib>
#include <sstream>

#include "gtn/gtn.h"
#include "parallel_compose.h"

using namespace gtn;

int main() {
  int M = 4;
  int N = 3;

  Graph g1 = linearGraph(M, 1);
  Graph g2 = linearGraph(M, 1);

  auto gOut = compose(g1, g2);
  std::cout << gOut << std::endl;
  std::cout << "NUM NODES " << gOut.numNodes() << std::endl;
//  std::cout << gOut << std::endl;
  auto gOutP = gtn::detail::dataparallel::compose(g1, g2);
  std::cout << gOutP << std::endl;
}
