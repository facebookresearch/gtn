/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gtn/gtn.h"

using namespace gtn;

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_rand, m) {
  m.def(
      "sample",
      [](const Graph& graph, size_t maxLength) {
        py::gil_scoped_release release;
        return sample(graph, maxLength);
      },
      "g"_a,
      "max_length"_a = 1000);
  m.def(
      "rand_equivalent",
      [](const Graph& g1,
         const Graph& g2,
         size_t numSamples,
         double tol,
         size_t maxLength) {
        py::gil_scoped_release release;
        return randEquivalent(g1, g2, numSamples, tol, maxLength);
      },
      "g1"_a,
      "g2"_a,
      "num_samples"_a = 1000,
      "tol"_a = 1e-4,
      "max_length"_a = 1000);
}
