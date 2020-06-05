#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

template <class T>
static T castBytes(const py::bytes& b) {
  static_assert(
      std::is_standard_layout<T>::value,
      "types represented as bytes must be standard layout");
  std::string s = b;
  if (s.size() != sizeof(T)) {
    throw std::runtime_error("wrong py::bytes size to represent object");
  }
  return *reinterpret_cast<const T*>(s.data());
}

namespace {
void ctcLoss(
    float* input,
    int numTimeSteps,
    int numFeatures,
    std::vector<std::vector<int>> labels,
    std::vector<float> lossScales,
    int blankIdx,
    float* loss,
    float* inputGrad) {
  bool computeGrad = (inputGrad != nullptr);
  int batchSize = labels.size();
  for (int b = 0; b < batchSize; ++b) {
    float* curInput = input + b * numTimeSteps * numFeatures;
    int* target = labels[b].data();
    // create emission graph
    Graph emissions(computeGrad);

    emissions.addNode(true);
    for (int t = 1; t <= numTimeSteps; ++t) {
      emissions.addNode(false, t == numTimeSteps);
      for (int n = 0; n < numFeatures; ++n) {
        emissions.addArc(t - 1, t, n, n, curInput[(t - 1) * numFeatures + n]);
      }
    }

    // create criterion graph
    Graph criterion(false);

    int L = labels[b].size();
    int S = 2 * L + 1;
    for (int l = 0; l < S; ++l) {
      int idx = (l - 1) / 2;
      criterion.addNode(l == 0, l == S - 1 || l == S - 2);
      int label = l % 2 ? target[idx] : blankIdx;
      criterion.addArc(l, l, label);
      if (l > 0) {
        criterion.addArc(l - 1, l, label);
      }
      if (l % 2 && l > 1 && label != target[idx - 1]) {
        criterion.addArc(l - 2, l, label);
      }
    }

    Graph fwdGraph =
        subtract(forward(emissions), forward(compose(emissions, criterion)));
    float scale = lossScales.size() > b ? lossScales[b] : 1.0;
    loss[b] = fwdGraph.item() * scale;
    if (computeGrad) {
      float* curInputGrad = inputGrad + b * numTimeSteps * numFeatures;
      backward(fwdGraph);
      for (int t = 0; t < numTimeSteps; ++t) {
        auto node = emissions.node(t);
        for (int n = 0; n < numFeatures; ++n) {
          curInputGrad[t * numFeatures + n] = node->out()[n]->grad() * scale;
        }
      }
    }
  }
}
}

PYBIND11_MODULE(_torch, m) {
  m.def(
      "ctc_loss",
      [](py::bytes input,
         int numTimeSteps,
         int numFeatures,
         std::vector<std::vector<int>> labels,
         std::vector<float> lossScales,
         int blankIdx,
         py::bytes loss,
         py::bytes inputGrad) {
        ctcLoss(
            castBytes<float*>(input),
            numTimeSteps,
            numFeatures,
            labels,
            lossScales,
            blankIdx,
            castBytes<float*>(loss),
            castBytes<float*>(inputGrad));
      });
}