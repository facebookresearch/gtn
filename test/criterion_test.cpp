#define CATCH_CONFIG_MAIN

#include <array>
#include <cmath>
#include <iostream>

#include "catch.hpp"

#include "gtn/autograd.h"
#include "gtn/functions.h"
#include "gtn/graph.h"
#include "gtn/utils.h"

using namespace gtn;

Graph emissions_graph(
    std::vector<float> emissions_vec,
    int T,
    int N,
    bool logprobs = false) {
  if (!logprobs) {
    std::transform(
        emissions_vec.begin(),
        emissions_vec.end(),
        emissions_vec.begin(),
        [](float p) -> float { return std::log(p); });
  }

  Graph emissions;
  emissions.addNode(true);
  for (int t = 1; t <= T; t++) {
    emissions.addNode(false, t == T);
    for (int n = 0; n < N; n++) {
      emissions.addArc(t - 1, t, n, n, emissions_vec[(t - 1) * N + n]);
    }
  }
  return emissions;
}

Graph ctc_graph(std::vector<int> target, int blank) {
  int L = target.size();
  int U = 2 * L + 1;
  Graph ctc;
  for (int l = 0; l < U; l++) {
    int idx = (l - 1) / 2;
    ctc.addNode(l == 0, l == U - 1 || l == U - 2);
    int label = l % 2 ? target[idx] : blank;
    ctc.addArc(l, l, label);
    if (l > 0) {
      ctc.addArc(l - 1, l, label);
    }
    if (l % 2 && l > 1 && label != target[idx - 1]) {
      ctc.addArc(l - 2, l, label);
    }
  }
  return ctc;
}

TEST_CASE("Test CTC", "[criterion.ctc]") {
  // These test cases are taken from wav2letter: https://fburl.com/msom2e4v
  {
    // Test case 1
    Graph ctc = ctc_graph({0, 0}, 1);

    Graph emissions = emissions_graph({1.0, 0.0, 0.0, 1.0, 1.0, 0.0}, 3, 2);

    auto loss = forwardScore(compose(ctc, emissions));
    CHECK(loss.item() == 0.0);

    // Should be 0 since scores are normalized
    auto z = forwardScore(emissions);
    CHECK(z.item() == 0.0);
  }

  {
    // Test case 2
    int T = 3, N = 4;
    Graph ctc = ctc_graph({1, 2}, N - 1);
    Graph emissions = emissions_graph(std::vector<float>(T * N, 1.0), T, N);

    auto expected_loss = -std::log(0.25 * 0.25 * 0.25 * 5);

    auto loss = subtract(forwardScore(compose(ctc, emissions)), forwardScore(emissions));
    CHECK(-loss.item() == Approx(expected_loss));
  }

  // This test case is  taken from Tensor Flow CTC implementation
  // tinyurl.com/y9du5v5a
  {
    // Test case 3
    const int T = 5, N = 6;
    std::vector<int> target = {0, 1, 2, 1, 0};

    // generate CTC graph
    Graph ctc = ctc_graph(target, N - 1);
    std::vector<float> emissions_vec = {
        0.633766,  0.221185, 0.0917319, 0.0129757,  0.0142857,  0.0260553,
        0.111121,  0.588392, 0.278779,  0.0055756,  0.00569609, 0.010436,
        0.0357786, 0.633813, 0.321418,  0.00249248, 0.00272882, 0.0037688,
        0.0663296, 0.643849, 0.280111,  0.00283995, 0.0035545,  0.00331533,
        0.458235,  0.396634, 0.123377,  0.00648837, 0.00903441, 0.00623107,
    };

    Graph emissions = emissions_graph(emissions_vec, T, N);

    // The log probabilities are already normalized,
    // so this should be close to 0
    auto z = forwardScore(emissions);
    CHECK(std::abs(z.item()) < 1e-5);

    auto loss = subtract(z, forwardScore(compose(ctc, emissions)));
    float expected_loss = 3.34211;
    CHECK(loss.item() == Approx(expected_loss));

    // Check the gradients
    backward(loss);

    std::array<float, N* T> expected_grad = {
        -0.366234, 0.221185,  0.0917319, 0.0129757,  0.0142857,  0.0260553,
        0.111121,  -0.411608, 0.278779,  0.0055756,  0.00569609, 0.010436,
        0.0357786, 0.633813,  -0.678582, 0.00249248, 0.00272882, 0.0037688,
        0.0663296, -0.356151, 0.280111,  0.00283995, 0.0035545,  0.00331533,
        -0.541765, 0.396634,  0.123377,  0.00648837, 0.00903441, 0.00623107};

    bool allClose = true;
    auto grad = emissions.grad();
    for (int i = 0; i < T * N; i++) {
      auto g = grad.weight(i);
      allClose &= (std::abs(expected_grad[i] - g) < 1e-5);
    }
    CHECK(allClose);
  }

  // This test case is  taken from Tensor Flow CTC implementation
  // tinyurl.com/y9du5v5a
  {
    // Test case 4
    const int T = 5, N = 6;
    std::vector<int> target = {0, 1, 1, 0};

    // generate CTC graph
    Graph ctc = ctc_graph(target, N - 1);

    std::vector<float> emissions_vec = {
        0.30176,  0.28562,  0.0831517, 0.0862751, 0.0816851, 0.161508,
        0.24082,  0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549,
        0.230246, 0.450868, 0.0389607, 0.038309,  0.0391602, 0.202456,
        0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345,
        0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046,
    };

    Graph emissions = emissions_graph(emissions_vec, T, N);

    // The log probabilities are already normalized,
    // so this should be close to 0
    auto z = forwardScore(emissions);
    CHECK(std::abs(z.item()) < 1e-5);

    auto loss = subtract(z, forwardScore(compose(ctc, emissions)));
    float expected_loss = 5.42262;
    CHECK(loss.item() == Approx(expected_loss));

    // Check the gradients
    backward(loss);

    std::array<float, N* T> expected_grad = {
        -0.69824,  0.28562,   0.0831517, 0.0862751, 0.0816851, 0.161508,
        0.24082,   -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549,
        0.230246,  0.450868,  0.0389607, 0.038309,  0.0391602, -0.797544,
        0.280884,  -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345,
        -0.576714, 0.315517,  0.0338439, 0.0393744, 0.0339315, 0.154046,
    };

    bool allClose = true;
    auto grad = emissions.grad();
    for (int i = 0; i < T * N; i++) {
      auto g = grad.weight(i);
      allClose &= (std::abs(expected_grad[i] - g) < 1e-5);
    }
    CHECK(allClose);
  }
}

TEST_CASE("Test ASG", "[criterion.asg]") {
  // This test cases is taken from wav2letter: https://fburl.com/msom2e4v
  const int T = 5, N = 6;

  std::vector<std::vector<int>> targets = {
      {2, 1, 5, 1, 3},
      {4, 3, 5},
      {3, 2, 2, 1},
  };

  std::vector<float> expected_loss = {
      7.7417464256287,
      6.4200420379639,
      8.2780694961548,
  };

  std::vector<std::vector<float>> emissions_vecs = {
      {-0.4340, -0.0254, 0.3667,  0.4180,  -0.3805, -0.1707, 0.1060, 0.3631,
       -0.1122, -0.3825, -0.0031, -0.3801, 0.0443,  -0.3795, 0.3194, -0.3130,
       0.0094,  0.1560,  0.1252,  0.2877,  0.1997,  -0.4554, 0.2774, -0.2526,
       -0.4001, -0.2402, 0.1295,  0.0172,  0.1805,  -0.3299},

      {
          0.3298,  -0.2259, -0.0959, 0.4909,  0.2996,  -0.2543,
          -0.2863, 0.3239,  -0.3988, 0.0732,  -0.2107, -0.4739,
          -0.0906, 0.0480,  -0.1301, 0.3975,  -0.3317, -0.1967,
          0.4372,  -0.2006, 0.0094,  0.3281,  0.1873,  -0.2945,
          0.2399,  0.0320,  -0.3768, -0.2849, -0.2248, 0.3186,
      },

      {
          0.0225,  -0.3867, -0.1929, -0.2904, -0.4958, -0.2533,
          0.4001,  -0.1517, -0.2799, -0.2915, 0.4198,  0.4506,
          0.1446,  -0.4753, -0.0711, 0.2876,  -0.1851, -0.1066,
          0.2081,  -0.1190, -0.3902, -0.1668, 0.1911,  -0.2848,
          -0.3846, 0.1175,  0.1052,  0.2172,  -0.0362, 0.3055,
      },
  };

  std::vector<std::array<float, N* T>> emissions_grads = {
      {
          0.1060, 0.1595,  -0.7639, 0.2485,  0.1118, 0.1380, 0.1915, -0.7524,
          0.1539, 0.1175,  0.1717,  0.1178,  0.1738, 0.1137, 0.2288, 0.1216,
          0.1678, -0.8057, 0.1766,  -0.7923, 0.1902, 0.0988, 0.2056, 0.1210,
          0.1212, 0.1422,  0.2059,  -0.8160, 0.2166, 0.1300,
      },

      {
          0.2029, 0.1164,  0.1325,  0.2383, -0.8032, 0.1131,  0.1414, 0.2602,
          0.1263, -0.3441, -0.3009, 0.1172, 0.1557,  0.1788,  0.1496, -0.5498,
          0.0140, 0.0516,  0.2306,  0.1219, 0.1503,  -0.4244, 0.1796, -0.2579,
          0.2149, 0.1745,  0.1160,  0.1271, 0.1350,  -0.7675,
      },

      {
          0.2195,  0.1458,  0.1770, -0.8395, 0.1307,  0.1666, 0.2148,  0.1237,
          -0.6613, -0.1223, 0.2191, 0.2259,  0.2002,  0.1077, -0.8386, 0.2310,
          0.1440,  0.1557,  0.2197, -0.1466, -0.5742, 0.1510, 0.2160,  0.1342,
          0.1050,  -0.8265, 0.1714, 0.1917,  0.1488,  0.2094,
      },
  };

  Graph transitions;
  for (int i = 0; i < N; i++) {
    transitions.addNode(true, true);
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      transitions.addArc(i, j, j); // p(j | i)
    }
  }
  for (size_t b = 0; b < targets.size(); b++) {
    auto target = targets[b];
    auto emissions_vec = emissions_vecs[b];
    auto emissions_grad = emissions_grads[b];

    Graph fal;
    fal.addNode(true);
    for (size_t l = 1; l <= target.size(); l++) {
      fal.addNode(false, l == target.size());
      fal.addArc(l - 1, l, target[l - 1]);
      fal.addArc(l, l, target[l - 1]);
    }

    Graph emissions = emissions_graph(emissions_vec, T, N, true);

    auto loss = subtract(
        forwardScore(compose(emissions, transitions)),
        forwardScore(compose(compose(fal, transitions), emissions)));

    CHECK(std::abs(loss.item() - expected_loss[b]) < 1e-3);

    // Check the gradients
    backward(loss);

    bool allClose = true;
    auto grad = emissions.grad();
    for (int i = 0; i < T * N; i++) {
      auto g = grad.weight(i);
      allClose &= (std::abs(emissions_grad[i] - g) < 1e-4);
    }
    CHECK(allClose);
  }

  bool allClose = true;
  std::array<float, N* N> trans_grad = {
      0.4871,  0.4369,  0.2711,  0.3106,  0.2931,  0.4336,
      0.4277,  0.0819,  0.2405, -0.7276,  0.2386, -0.6247,
      0.4366, -1.5975, -1.2340,  0.2459,  0.2513,  0.3684,
      0.4802,  0.4439, -0.7560, -0.9118,  0.2729, -0.6026,
      0.4385,  0.4064,  0.2459, -0.7159, -0.3097,  0.3911,
      0.4036, -0.6449,  0.1965,  0.2282,  0.2105, -0.1164
  };

  auto grad = transitions.grad();
  for (int i = 0; i < N * N; i++) {
    auto g = grad.weight(i);
    allClose &= (std::abs(trans_grad[i] - g) < 1e-4);
  }
  CHECK(allClose);
}
