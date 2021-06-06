/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>
#include <tuple>
#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "parallel_compose.h"
#include "prefix_scan.h"

namespace gtn {
namespace detail {
namespace dataparallel {

namespace {

struct GraphDataParallelGPU {
  size_t numNodes;
  size_t numArcs;

  // True if a node is accept or start, false otherwise
  int* accept;
  int* start;

  // One value per node - i-th value corresponds to i-th node
  // Last element is the total number of arcs, so that
  // each element and its neighbor forms a range
  int* inArcOffset;
  int* outArcOffset;

  // One value per arc
  int* inArcs;
  int* outArcs;

  // One value per arc
  // i-th value corresponds to i-th arc
  int* ilabels;
  int* olabels;
  int* srcNodes;
  int* dstNodes;
  float* weights;
};

struct nodeAndArcPairGPU {
  int2 nodePair;
  int2 arcPair;
  int2 checkEpsilonArcPair;
  bool checkArcPair;
  bool isValid;
};

inline int div_up(int x, int y) {
  return (x + y - 1) / y;
}

__device__ __host__
inline int TwoDToOneDIndex(int n1, int n2, int n1Extent) {
  assert(n1 < n1Extent);
  return n1 + n2 * n1Extent;
}

__device__
inline int2 OneDToTwoDIndexGPU(int n, int n1Extent) {
  assert(n1Extent > 0);
  const int n2 = n / n1Extent;
  const int n1 = n % n1Extent;
  return make_int2(n1, n2);
}

bool checkAnyTrueGPU(const int* flags, int numFlags) {
  thrust::device_ptr<const int> tPtr(flags);
  const int sum = thrust::reduce(tPtr, tPtr + numFlags, int(0));

  return (sum > 0);
}

std::tuple<int*, size_t, int> prefixSumScanGPU(const int* input, size_t numElts, bool appendSum) {
  const size_t scanNumElts = appendSum ? numElts + 1 : numElts;

  int *output;
  cudaMalloc((void **)(&(output)), sizeof(int) * scanNumElts);
  cudaMemcpy((void *)(output), (void *)(input), sizeof(int) * numElts, cudaMemcpyDeviceToDevice);

  int sum = 0;
  if (numElts > 0) {
    thrust::device_ptr<int> tPtr(output);
    thrust::exclusive_scan(tPtr, tPtr + numElts, tPtr);

    int lastElementInput;
    int lastElementOutput;
    cudaMemcpy((void *)(&lastElementInput), (void *)(&(input[numElts-1])), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)(&lastElementOutput), (void *)(&(output[numElts-1])), sizeof(int), cudaMemcpyDeviceToHost);
    sum = lastElementInput + lastElementOutput;
  }

  if (appendSum) {
    assert(scanNumElts > 0);
    cudaMemcpy((void *)(&(output[scanNumElts-1])), (void *)(&sum), sizeof(int), cudaMemcpyHostToDevice);
  }

  return std::make_tuple(output, scanNumElts, sum);
}


// Map thread id to corresponding node and arc pair
// Also map thread id to two flags checkEpsilonArcPair.first,
// checkEpsilonArcPair.second When checkEpsilonArcPair.first is set,
// corresponding tid will check for arcs with epsilon arcs in the node from
// first graph Same logic happens for checkEpsilonArcPair.second Search to find
// which node pair this tid will fall into Linear search for now
// (arcCrossProductOffset is sorted by definition)
__device__
nodeAndArcPairGPU computeNodeAndArcPair(
    int tid,
    size_t numArcCrossProductOffset,
    const int* arcCrossProductOffset,
    const int* toExploreNumArcsFirst,
    const int* toExploreNumArcsSecond,
    const int* toExploreNodePairFirst,
    const int* toExploreNodePairSecond) {

  nodeAndArcPairGPU result;
  result.checkArcPair = false;
  result.checkEpsilonArcPair = make_int2(false, false);
  result.isValid = false;

  // There should be at least two values to form a range
  assert(numArcCrossProductOffset >= 2);

  for (size_t i = 0; i < numArcCrossProductOffset - 1; ++i) {
    const int lVal = arcCrossProductOffset[i];
    const int rVal = arcCrossProductOffset[i + 1];

    if ((lVal <= tid) && (tid < rVal)) {
      result.isValid = true;
      result.nodePair = make_int2(
          toExploreNodePairFirst[i], toExploreNodePairSecond[i]);

      // The range of idx is from
      // [0, toExploreNumArcsFirst[i] * toExploreNumArcsSecond[i])
      const int idx = tid - lVal;
      const int numArcs = rVal - lVal;

      assert(idx >= 0);
      assert(idx < numArcs);
      assert(numArcs > 0);

      const int arcProd =
          toExploreNumArcsFirst[i] * toExploreNumArcsSecond[i];

      if (numArcs == arcProd) {
        result.checkArcPair = true;

        // We map the tids to 2D grid where the
        // x-axis is toExploreNumArcsFirst[i] (row)
        // y-axis is toExploreNumArcsSecond[i] (column)
	assert(toExploreNumArcsFirst[i] > 0);
        result.arcPair = make_int2(
	    idx % toExploreNumArcsFirst[i],
	    idx / toExploreNumArcsFirst[i]);

        // Pick the tids from the first row since we need only one
        // tid per arc of the node from the first graph to check for
        // epsilon
        if (idx < toExploreNumArcsFirst[i]) {
          result.checkEpsilonArcPair.x = true;
        }

        // Pick the tids from the first column since we need only one
        // tid per arc of the node from the first graph to check for
        // epsilon
        if ((idx % toExploreNumArcsFirst[i]) == 0) {
          result.checkEpsilonArcPair.y = true;
        }
      } else if ((arcProd == 0) && (numArcs == toExploreNumArcsFirst[i])) {
        // TODO: Likely not the brightest idea to use -1 as sentinel
        result.arcPair = make_int2(idx, -1);
        result.checkEpsilonArcPair.x = true;
      } else if ((arcProd == 0) && (numArcs == toExploreNumArcsSecond[i])) {
        // TODO: Likely not the brightest idea to use -1 as sentinel
        result.arcPair = make_int2(-1, idx);
        result.checkEpsilonArcPair.y = true;
      }

      break;
    }
  }

  return result;
}

__global__
void calculateArcCrossProductOffsetKernel(
      const GraphDataParallelGPU graphDP1GPU,
      const GraphDataParallelGPU graphDP2GPU,
      const int* toExploreNodePairFirstGPU,
      const int* toExploreNodePairSecondGPU,
      int* toExploreNumArcsFirstGPU,
      int* toExploreNumArcsSecondGPU,
      int* arcCrossProductOffsetGPU,
      size_t numToExploreNodePair,
      bool inOrOutArc) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numToExploreNodePair) {
    int node = toExploreNodePairFirstGPU[gTid];
    // Special case if it is the last node. Then the offset becomes
    // the number of arcs
    const int inArcOffsetGraph1 = ((node + 1) == graphDP1GPU.numNodes)
        ? graphDP1GPU.numArcs
        : graphDP1GPU.inArcOffset[node + 1];
    const int outArcOffsetGraph1 = ((node + 1) == graphDP1GPU.numNodes)
        ? graphDP1GPU.numArcs
        : graphDP1GPU.outArcOffset[node + 1];

    const int numArcsFirst = inOrOutArc
        ? inArcOffsetGraph1 - graphDP1GPU.inArcOffset[node]
        : outArcOffsetGraph1 - graphDP1GPU.outArcOffset[node];

    node = toExploreNodePairSecondGPU[gTid];
    // Special case if it is the last node. Then the offset becomes
    // the number of arcs
    const int inArcOffsetGraph2 = ((node + 1) == graphDP2GPU.numNodes)
        ? graphDP2GPU.numArcs
        : graphDP2GPU.inArcOffset[node + 1];
    const int outArcOffsetGraph2 = ((node + 1) == graphDP2GPU.numNodes)
        ? graphDP2GPU.numArcs
        : graphDP2GPU.outArcOffset[node + 1];

    const int numArcsSecond = inOrOutArc
        ? inArcOffsetGraph2 - graphDP2GPU.inArcOffset[node]
        : outArcOffsetGraph2 - graphDP2GPU.outArcOffset[node];

    toExploreNumArcsFirstGPU[gTid] = numArcsFirst;
    toExploreNumArcsSecondGPU[gTid] = numArcsSecond;

    // Even when numArcsFirst or numArcsSecond is 0 we have to consider
    // the case when the other graph has arcs with epsilon label
    if (numArcsFirst != 0 && numArcsSecond != 0) {
      arcCrossProductOffsetGPU[gTid] = numArcsFirst * numArcsSecond;
    } else if (numArcsFirst != 0 && numArcsSecond == 0) {
      arcCrossProductOffsetGPU[gTid] = numArcsFirst;
    } else if (numArcsFirst == 0 && numArcsSecond != 0) {
      arcCrossProductOffsetGPU[gTid] = numArcsSecond;
    } else {
      arcCrossProductOffsetGPU[gTid] = 0;
    }
  }
}

// Takes a pair of nodes, where each member of pair comes from a different
// graph and calculate a vector of number of arcs in the cross product of
// arcs outgoing from each pair.
// This should be a kernel call
std::tuple<int*, int*, int*>
calculateArcCrossProductOffsetGPU(
    const int* toExploreNodePairFirstGPU,
    const int* toExploreNodePairSecondGPU,
    size_t numToExploreNodePair,
    const GraphDataParallelGPU graphDP1GPU,
    const GraphDataParallelGPU graphDP2GPU,
    bool inOrOutArc) {

  int* toExploreNumArcsFirstGPU;
  int* toExploreNumArcsSecondGPU;
  int* arcCrossProductOffsetGPU;
  cudaMalloc((void **)(&(toExploreNumArcsFirstGPU)), sizeof(int) * numToExploreNodePair);
  cudaMalloc((void **)(&(toExploreNumArcsSecondGPU)), sizeof(int) * numToExploreNodePair);
  cudaMalloc((void **)(&(arcCrossProductOffsetGPU)), sizeof(int) * numToExploreNodePair);

  const int NT = 128;
  const int gridSize = div_up(numToExploreNodePair, NT);

  calculateArcCrossProductOffsetKernel<<<gridSize, NT, 0, 0>>>(
      graphDP1GPU, graphDP2GPU, toExploreNodePairFirstGPU, toExploreNodePairSecondGPU,
      toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU, arcCrossProductOffsetGPU,
      numToExploreNodePair, inOrOutArc);

  return std::make_tuple(arcCrossProductOffsetGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU);
}

// This function needs to be thread safe since multiple threads can
// can call it and they will overlap on curIdx and dstIdx
__device__
void calculateNumArcsAndNodesToExplore(
    int curIdx,
    int dstIdx,
    const int* reachable,
    int* newNodes,
    int* toExplore,
    int* numOutArcs,
    int* numInArcs) {
  if (reachable[dstIdx]) {
    // Atomic test and set for newNodes
    /*
    int oldVal = newNodes[dstIdx];
    if (!newNodes[dstIdx]) {
      newNodes[dstIdx] = true;
    }*/
    int oldVal = atomicCAS(&(newNodes[dstIdx]), false, true);
    if (!oldVal) {
      toExplore[dstIdx] = true;
    }

    // These are atomic increments
    // numOutArcs[curIdx]++;
    // numInArcs[dstIdx]++;
    atomicAdd(&(numOutArcs[curIdx]), 1);
    atomicAdd(&(numInArcs[dstIdx]), 1);

    // printf("cidx %d didx %d\n", curIdx, dstIdx);
    // printf("no %d ni %d\n", numOutArcs[curIdx], numInArcs[dstIdx]);
  }
}

// This function needs to be thread safe since multiple threads can
// can call it
__device__
void generateCombinedGraphNodesAndArcs(
    int dstIdx,
    int curIdx,
    const int2& arcPair,
    const int2& dstNodeStartAndAccept,
    const int* reachable,
    const int* newNodesOffset,
    int* newNodesVisited,
    int* toExplore,
    int* gradInfoFirst,
    int* gradInfoSecond,
    GraphDataParallelGPU& newGraphDP,
    int ilabel,
    int olabel,
    float weight) {
  if (reachable[dstIdx]) {
    // Atomic test and set for newNodesVisited
    /*
    int oldVal = newNodesVisited[dstIdx];
    if (!newNodesVisited[dstIdx]) {
      newNodesVisited[dstIdx] = true;
    }*/

    int oldVal = atomicCAS(&(newNodesVisited[dstIdx]), false, true);
    if (!oldVal) {
      toExplore[dstIdx] = true;
    }

    // Set accept and start nodes
    // I think I only need it for dst nodes and src nodes
    // Note: Multiple threads can have the same dstIdx and write to the same
    //       location and collide. This _should_ be fine since they are going
    //       to write the same value
    newGraphDP.start[newNodesOffset[dstIdx]] = dstNodeStartAndAccept.x;
    newGraphDP.accept[newNodesOffset[dstIdx]] = dstNodeStartAndAccept.y;

    // Both of these increments are atomic
    // int inArcIdx = newGraphDP.inArcOffset[newNodesOffset[dstIdx]]++;
    // int outArcIdx = newGraphDP.outArcOffset[newNodesOffset[curIdx]]++;

    int inArcIdx = atomicAdd(&(newGraphDP.inArcOffset[newNodesOffset[dstIdx]]), 1);
    int outArcIdx = atomicAdd(&(newGraphDP.outArcOffset[newNodesOffset[curIdx]]), 1);

    // printf("dstIdx %d curIdx %d\n", dstIdx, curIdx);
    // printf("inArcIdx %d outArcIdx %d\n", inArcIdx, outArcIdx);

    // outArcIdx is also the arc identifier
    newGraphDP.outArcs[outArcIdx] = outArcIdx;
    newGraphDP.inArcs[inArcIdx] = outArcIdx;

    // Fill in everything else for this arc
    newGraphDP.ilabels[outArcIdx] = ilabel;
    newGraphDP.olabels[outArcIdx] = olabel;
    newGraphDP.srcNodes[outArcIdx] = newNodesOffset[curIdx];
    newGraphDP.dstNodes[outArcIdx] = newNodesOffset[dstIdx];
    newGraphDP.weights[outArcIdx] = weight;

    // printf("ilabels %d olabels %d srcNodes %d dstNodes %d weights %f\n",
           // newGraphDP.ilabels[outArcIdx], newGraphDP.olabels[outArcIdx],
	   // newGraphDP.srcNodes[outArcIdx], newGraphDP.dstNodes[outArcIdx],
	   // newGraphDP.weights[outArcIdx]);

    gradInfoFirst[outArcIdx] = arcPair.x;
    gradInfoSecond[outArcIdx] = arcPair.y;
  }
}

__global__
void convertToNodePairKernel(
  const int* flagsGPU,
  const int* indicesGPU,
  int* toExploreNodePairFirstGPU,
  int* toExploreNodePairSecondGPU,
  int extent,
  size_t numFlags,
  size_t numValidNodes) {

  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numFlags) {
    if (flagsGPU[gTid] == true) {
      const int index = indicesGPU[gTid];
      assert(index >= 0);
      assert(index < numValidNodes);

      int2 node = OneDToTwoDIndexGPU(gTid, extent);
      toExploreNodePairFirstGPU[index] = node.x;
      toExploreNodePairSecondGPU[index] = node.y;
    }
  }
}

// Convert bool array two pairs for true flags
std::tuple<int*, int*, size_t> convertToNodePairGPU(
    const int* flagsGPU,
    size_t numFlags,
    int extent) {
  int* indicesGPU;
  size_t numIndices;
  size_t numValidNodes;

  std::tie(indicesGPU, numIndices, numValidNodes) = prefixSumScanGPU(flagsGPU, numFlags, false);
  assert(numFlags == numIndices);

  int* toExploreNodePairFirstGPU;
  int* toExploreNodePairSecondGPU;
  cudaMalloc((void **)(&(toExploreNodePairFirstGPU)), sizeof(int) * numValidNodes);
  cudaMalloc((void **)(&(toExploreNodePairSecondGPU)), sizeof(int) * numValidNodes);

  const int NT = 128;
  const int gridSize = div_up(numFlags, NT);

  convertToNodePairKernel<<<gridSize, NT, 0, 0>>>(flagsGPU, indicesGPU,
    toExploreNodePairFirstGPU, toExploreNodePairSecondGPU,
    extent, numFlags, numValidNodes);

  cudaFree(indicesGPU);
  return std::make_tuple(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, numValidNodes);
}

__device__
int2 getStartAndAccept(
    const GraphDataParallelGPU& graphDP1,
    const GraphDataParallelGPU& graphDP2,
    const int2& dstNodePair) {

  int2 dstNodeStartAndAccept = make_int2(
      graphDP1.start[dstNodePair.x] && graphDP2.start[dstNodePair.y],
      graphDP1.accept[dstNodePair.x] &&
          graphDP2.accept[dstNodePair.y]);

  return dstNodeStartAndAccept;
}

GraphDataParallelGPU copyToGPU(const GraphDataParallel& graphDP) {
  GraphDataParallelGPU graphDPGPU;

  graphDPGPU.numNodes = graphDP.inArcOffset.size();
  graphDPGPU.numArcs = graphDP.inArcs.size();

  assert(graphDP.accept.size() == graphDPGPU.numNodes);
  assert(graphDP.start.size() == graphDPGPU.numNodes);
  assert(graphDP.inArcOffset.size() == graphDPGPU.numNodes);
  assert(graphDP.outArcOffset.size() == graphDPGPU.numNodes);

  assert(graphDP.inArcs.size() == graphDPGPU.numArcs);
  assert(graphDP.outArcs.size() == graphDPGPU.numArcs);
  assert(graphDP.ilabels.size() == graphDPGPU.numArcs);
  assert(graphDP.olabels.size() == graphDPGPU.numArcs);
  assert(graphDP.srcNodes.size() == graphDPGPU.numArcs);
  assert(graphDP.dstNodes.size() == graphDPGPU.numArcs);
  assert(graphDP.weights.size() == graphDPGPU.numArcs);

  // Allocate memory
  cudaMalloc((void **)(&(graphDPGPU.accept)), sizeof(int) * graphDPGPU.numNodes);

  cudaMalloc((void **)(&(graphDPGPU.start)), sizeof(int) * graphDPGPU.numNodes);

  cudaMalloc((void **)(&(graphDPGPU.inArcOffset)), sizeof(int) * graphDPGPU.numNodes);
  cudaMalloc((void **)(&(graphDPGPU.outArcOffset)), sizeof(int) * graphDPGPU.numNodes);

  cudaMalloc((void **)(&(graphDPGPU.inArcs)), sizeof(int) * graphDPGPU.numArcs);
  cudaMalloc((void **)(&(graphDPGPU.outArcs)), sizeof(int) * graphDPGPU.numArcs);

  cudaMalloc((void **)(&(graphDPGPU.ilabels)), sizeof(int) * graphDPGPU.numArcs);
  cudaMalloc((void **)(&(graphDPGPU.olabels)), sizeof(int) * graphDPGPU.numArcs);
  cudaMalloc((void **)(&(graphDPGPU.srcNodes)), sizeof(int) * graphDPGPU.numArcs);
  cudaMalloc((void **)(&(graphDPGPU.dstNodes)), sizeof(int) * graphDPGPU.numArcs);
  cudaMalloc((void **)(&(graphDPGPU.weights)), sizeof(float) * graphDPGPU.numArcs);

  // Copy
  cudaMemcpy((void *)(graphDPGPU.accept), (void *)(graphDP.accept.data()), sizeof(int) * graphDPGPU.numNodes, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)(graphDPGPU.start), (void *)(graphDP.start.data()), sizeof(int) * graphDPGPU.numNodes, cudaMemcpyHostToDevice);

  cudaMemcpy((void *)(graphDPGPU.inArcOffset), (void *)(graphDP.inArcOffset.data()), sizeof(int) * graphDPGPU.numNodes, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)(graphDPGPU.outArcOffset), (void *)(graphDP.outArcOffset.data()), sizeof(int) * graphDPGPU.numNodes, cudaMemcpyHostToDevice);

  cudaMemcpy((void *)(graphDPGPU.inArcs), (void *)(graphDP.inArcs.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)(graphDPGPU.outArcs), (void *)(graphDP.outArcs.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice);

  cudaMemcpy((void *)(graphDPGPU.ilabels), (void *)(graphDP.ilabels.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)(graphDPGPU.olabels), (void *)(graphDP.olabels.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)(graphDPGPU.srcNodes), (void *)(graphDP.srcNodes.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)(graphDPGPU.dstNodes), (void *)(graphDP.dstNodes.data()), sizeof(int) * graphDPGPU.numArcs, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)(graphDPGPU.weights), (void *)(graphDP.weights.data()), sizeof(float) * graphDPGPU.numArcs, cudaMemcpyHostToDevice);

  return graphDPGPU;
}

__global__ 
void findReachableKernel(
      const GraphDataParallelGPU graphDP1GPU,
      const GraphDataParallelGPU graphDP2GPU,
      const int* arcCrossProductOffsetGPU,
      const int* toExploreNumArcsFirstGPU,
      const int* toExploreNumArcsSecondGPU,
      const int* toExploreNodePairFirstGPU,
      const int* toExploreNodePairSecondGPU,
      int numNodesFirst,
      int totalArcs,
      size_t numArcCrossProductOffset,
      int* toExploreGPU,
      int* reachableGPU,
      int* epsilonMatchedGPU
      ) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    nodeAndArcPairGPU result = 
      computeNodeAndArcPair(
        gTid, numArcCrossProductOffset, arcCrossProductOffsetGPU,
        toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU,
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU);

    // printf("tid = %d, valid = %d\n", gTid, result.isValid);
    // Does this node pair match?
    if (result.isValid) {

      int inArcOffset = graphDP1GPU.inArcOffset[result.nodePair.x];
      const int firstArcIdx = graphDP1GPU.inArcs[inArcOffset + result.arcPair.x];

      inArcOffset = graphDP2GPU.inArcOffset[result.nodePair.y];
      const int secondArcIdx = graphDP2GPU.inArcs[inArcOffset + result.arcPair.y];

      // printf("tid = %d, cp = %d\n", gTid, result.checkArcPair);

      if (result.checkArcPair &&
          (graphDP1GPU.olabels[firstArcIdx] == graphDP2GPU.ilabels[secondArcIdx])) {
        const int idx = TwoDToOneDIndex(
            graphDP1GPU.srcNodes[firstArcIdx],
            graphDP2GPU.srcNodes[secondArcIdx],
            numNodesFirst);

	// printf("tid = %d, idx = %d\n", gTid, idx);

        if (graphDP1GPU.olabels[firstArcIdx] == epsilon) {
          epsilonMatchedGPU[idx] = true;
        }

        // idx may not be unique amongst all threads.
        /*
        int oldVal = reachableGPU[idx];
        if (!reachableGPU[idx]) {
          reachableGPU[idx] = true;
        }*/
        int oldVal = atomicCAS(&(reachableGPU[idx]), false, true);
        if (!oldVal) {
          toExploreGPU[idx] = true;
        }
	// printf("r %d t %d \n", reachableGPU[idx], toExploreGPU[idx]);
      }

      // Only valid for arcs incoming to node from first graph
      if (result.checkEpsilonArcPair.x &&
          (graphDP1GPU.olabels[firstArcIdx] == epsilon)) {
        const int idx = TwoDToOneDIndex(
            graphDP1GPU.srcNodes[firstArcIdx], result.nodePair.y, numNodesFirst);
        /*
        int oldVal = reachableGPU[idx];
        if (!reachableGPU[idx]) {
          reachableGPU[idx] = true;
        }*/
        int oldVal = atomicCAS(&(reachableGPU[idx]), false, true);
        if (!oldVal) {
          toExploreGPU[idx] = true;
        }
      }

      // Only valid for arcs incoming to node from second graph
      if (result.checkEpsilonArcPair.y &&
          (graphDP2GPU.ilabels[secondArcIdx] == epsilon)) {
        const int idx = TwoDToOneDIndex(
            result.nodePair.x, graphDP2GPU.srcNodes[secondArcIdx], numNodesFirst);
        /*
        int oldVal = reachableGPU[idx];
        if (!reachableGPU[idx]) {
          reachableGPU[idx] = true;
        }*/
        int oldVal = atomicCAS(&(reachableGPU[idx]), false, true);
        if (!oldVal) {
          toExploreGPU[idx] = true;
        }
      }
    }
  }
}

__global__ 
void computeValidNodeAndArcKernel(
      const GraphDataParallelGPU graphDP1GPU,
      const GraphDataParallelGPU graphDP2GPU,
      const int* arcCrossProductOffsetGPU,
      const int* toExploreNumArcsFirstGPU,
      const int* toExploreNumArcsSecondGPU,
      const int* toExploreNodePairFirstGPU,
      const int* toExploreNodePairSecondGPU,
      const int* reachableGPU,
      const int* epsilonMatchedGPU,
      int numNodesFirst,
      int totalArcs,
      size_t numArcCrossProductOffset,
      int* toExploreGPU,
      int* newNodesGPU,
      int* numInArcsGPU,
      int* numOutArcsGPU
      ) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    // Map tid to corresponding node and arc pair
    // Search to find which node pair this tid will fall into
    nodeAndArcPairGPU result = 
      computeNodeAndArcPair(
        gTid, numArcCrossProductOffset, arcCrossProductOffsetGPU,
        toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU,
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU);

    if (result.isValid) {
      int outArcOffset = graphDP1GPU.outArcOffset[result.nodePair.x];
      const int firstArcIdx = graphDP1GPU.outArcs[outArcOffset + result.arcPair.x];

      outArcOffset = graphDP2GPU.outArcOffset[result.nodePair.y];
      const int secondArcIdx =
          graphDP2GPU.outArcs[outArcOffset + result.arcPair.y];

      const bool epsilonMatch = epsilonMatchedGPU[TwoDToOneDIndex(
          result.nodePair.x, result.nodePair.y, numNodesFirst)];

      // Does this node pair match?
      // Skip epsilon matches
      if (result.checkArcPair &&
          (graphDP1GPU.olabels[firstArcIdx] == graphDP2GPU.ilabels[secondArcIdx])) {
        const int dstIdx = TwoDToOneDIndex(
            graphDP1GPU.dstNodes[firstArcIdx],
            graphDP2GPU.dstNodes[secondArcIdx],
            numNodesFirst);
        const int curIdx =
            TwoDToOneDIndex(result.nodePair.x, result.nodePair.y, numNodesFirst);

        // printf("krnl 1a dst %d cur %d\n", dstIdx, curIdx);

        // We track if any two arcs outgoing from this node pair match
        // on epsilon. We record if they do.
        if (graphDP1GPU.olabels[firstArcIdx] != epsilon) {
          calculateNumArcsAndNodesToExplore(
              curIdx,
              dstIdx,
              reachableGPU,
              newNodesGPU,
              toExploreGPU,
              numOutArcsGPU,
              numInArcsGPU);
        }
      }

      if (result.checkEpsilonArcPair.x &&
          (!epsilonMatch || graphDP2GPU.accept[result.nodePair.y] ||
           !graphDP1GPU.accept[result.nodePair.x]) &&
          (graphDP1GPU.olabels[firstArcIdx] == epsilon)) {
        const int dstIdx = TwoDToOneDIndex(
            graphDP1GPU.dstNodes[firstArcIdx], result.nodePair.y, numNodesFirst);
        const int curIdx =
            TwoDToOneDIndex(result.nodePair.x, result.nodePair.y, numNodesFirst);

        // printf("krnl 1b dst %d cur %d\n", dstIdx, curIdx);

        calculateNumArcsAndNodesToExplore(
            curIdx,
            dstIdx,
            reachableGPU,
            newNodesGPU,
            toExploreGPU,
            numOutArcsGPU,
            numInArcsGPU);
      }

      if (result.checkEpsilonArcPair.y &&
          (!epsilonMatch || graphDP1GPU.accept[result.nodePair.x]) &&
          (graphDP2GPU.ilabels[secondArcIdx] == epsilon)) {
        const int dstIdx = TwoDToOneDIndex(
            result.nodePair.x, graphDP2GPU.dstNodes[secondArcIdx], numNodesFirst);
        const int curIdx =
            TwoDToOneDIndex(result.nodePair.x, result.nodePair.y, numNodesFirst);

        // printf("krnl 1c dst %d cur %d\n", dstIdx, curIdx);

        calculateNumArcsAndNodesToExplore(
            curIdx,
            dstIdx,
            reachableGPU,
            newNodesGPU,
            toExploreGPU,
            numOutArcsGPU,
            numInArcsGPU);
      }
    }
  }
}

__global__ 
void generateNodeAndArcKernel(
      const GraphDataParallelGPU graphDP1GPU,
      const GraphDataParallelGPU graphDP2GPU,
      const int* arcCrossProductOffsetGPU,
      const int* toExploreNumArcsFirstGPU,
      const int* toExploreNumArcsSecondGPU,
      const int* toExploreNodePairFirstGPU,
      const int* toExploreNodePairSecondGPU,
      const int* reachableGPU,
      const int* epsilonMatchedGPU,
      int numNodesFirst,
      int totalArcs,
      size_t numArcCrossProductOffset,
      GraphDataParallelGPU newGraphDPGPU,
      int* toExploreGPU,
      int* gradInfoFirstGPU,
      int* gradInfoSecondGPU,
      int* newNodesOffsetGPU,
      int* newNodesVisitedGPU
      ) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < totalArcs) {
    // Map tid to corresponding node and arc pair
    // Search to find which node pair this tid will fall into
    nodeAndArcPairGPU result = 
      computeNodeAndArcPair(
        gTid, numArcCrossProductOffset, arcCrossProductOffsetGPU,
        toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU,
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU);

    if (result.isValid) {
      int outArcOffset = graphDP1GPU.outArcOffset[result.nodePair.x];
      const int firstArcIdx = graphDP1GPU.outArcs[outArcOffset + result.arcPair.x];

      outArcOffset = graphDP2GPU.outArcOffset[result.nodePair.y];
      const int secondArcIdx =
          graphDP2GPU.outArcs[outArcOffset + result.arcPair.y];

      const bool epsilonMatch = epsilonMatchedGPU[TwoDToOneDIndex(
          result.nodePair.x, result.nodePair.y, numNodesFirst)];

      // Does this node pair match?
      if (result.checkArcPair &&
          (graphDP1GPU.olabels[firstArcIdx] == graphDP2GPU.ilabels[secondArcIdx])) {
        int2 dstNodePair = make_int2(
            graphDP1GPU.dstNodes[firstArcIdx], graphDP2GPU.dstNodes[secondArcIdx]);

        const int dstIdx = TwoDToOneDIndex(
            dstNodePair.x, dstNodePair.y, numNodesFirst);
        const int curIdx = TwoDToOneDIndex(
            result.nodePair.x, result.nodePair.y, numNodesFirst);

	// printf("krn2a dstIdx=%d curIdx=%d\n", dstIdx, curIdx);

        const int2 dstNodeStartAccept =
            getStartAndAccept(graphDP1GPU, graphDP2GPU, dstNodePair);

        // We track if any two arcs outgoing from this node pair match
        // on epsilon. We record if they do.
        if (graphDP1GPU.olabels[firstArcIdx] != epsilon) {
          generateCombinedGraphNodesAndArcs(
              dstIdx,
              curIdx,
              make_int2(firstArcIdx, secondArcIdx),
              dstNodeStartAccept,
              reachableGPU,
              newNodesOffsetGPU,
              newNodesVisitedGPU,
              toExploreGPU,
              gradInfoFirstGPU,
              gradInfoSecondGPU,
              newGraphDPGPU,
              graphDP1GPU.ilabels[firstArcIdx],
              graphDP2GPU.olabels[secondArcIdx],
              graphDP1GPU.weights[firstArcIdx] + graphDP2GPU.weights[secondArcIdx]);
        }
      }

      // The epsilon matches
      if (result.checkEpsilonArcPair.x &&
          (!epsilonMatch || graphDP2GPU.accept[result.nodePair.y] ||
           !graphDP1GPU.accept[result.nodePair.x]) &&
          (graphDP1GPU.olabels[firstArcIdx] == epsilon)) {
        // When arc from first node has epsilon label then we consider
        // second node
        int2 dstNodePair = make_int2(
            graphDP1GPU.dstNodes[firstArcIdx], result.nodePair.y);
        const int dstIdx = TwoDToOneDIndex(
            dstNodePair.x, dstNodePair.y, numNodesFirst);
        const int curIdx = TwoDToOneDIndex(
            result.nodePair.x, result.nodePair.y, numNodesFirst);

	// printf("krn2b dstIdx=%d curIdx=%d\n", dstIdx, curIdx);

        const int2 dstNodeStartAccept =
            getStartAndAccept(graphDP1GPU, graphDP2GPU, dstNodePair);

        generateCombinedGraphNodesAndArcs(
            dstIdx,
            curIdx,
            make_int2(firstArcIdx, -1),
            dstNodeStartAccept,
            reachableGPU,
            newNodesOffsetGPU,
            newNodesVisitedGPU,
            toExploreGPU,
            gradInfoFirstGPU,
            gradInfoSecondGPU,
            newGraphDPGPU,
            graphDP1GPU.ilabels[firstArcIdx],
            epsilon,
            graphDP1GPU.weights[firstArcIdx]);
      }

      // The epsilon matches
      if (result.checkEpsilonArcPair.y &&
          (!epsilonMatch || graphDP1GPU.accept[result.nodePair.x]) &&
          (graphDP2GPU.ilabels[secondArcIdx] == epsilon)) {
        // When arc from second node has epsilon label then we consider
        // first node
        int2 dstNodePair = make_int2(
            result.nodePair.x, graphDP2GPU.dstNodes[secondArcIdx]);
        const int dstIdx = TwoDToOneDIndex(
            dstNodePair.x, dstNodePair.y, numNodesFirst);
        const int curIdx = TwoDToOneDIndex(
            result.nodePair.x, result.nodePair.y, numNodesFirst);

	// printf("krn2c dstIdx=%d curIdx=%d\n", dstIdx, curIdx);
	
        const int2 dstNodeStartAndAccept =
            getStartAndAccept(graphDP1GPU, graphDP2GPU, dstNodePair);

        generateCombinedGraphNodesAndArcs(
            dstIdx,
            curIdx,
            make_int2(-1, secondArcIdx),
            dstNodeStartAndAccept,
            reachableGPU,
            newNodesOffsetGPU,
            newNodesVisitedGPU,
            toExploreGPU,
            gradInfoFirstGPU,
            gradInfoSecondGPU,
            newGraphDPGPU,
            epsilon,
            graphDP2GPU.olabels[secondArcIdx],
            graphDP2GPU.weights[secondArcIdx]);
      }
    }
  }
}

__global__
void calculateNumArcsKernel(
  const int* flagsGPU,
  const int* indicesGPU,
  const int* inputInArcsGPU,
  const int* inputOutArcsGPU,
  int* outputInArcsGPU,
  int* outputOutArcsGPU,
  size_t numFlags,
  size_t numValidNodes) {

  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numFlags) {
    if (flagsGPU[gTid] == true) {
      const int index = indicesGPU[gTid];
      assert(index >= 0);
      assert(index < numValidNodes);

      outputInArcsGPU[index] = inputInArcsGPU[gTid];
      outputOutArcsGPU[index] = inputOutArcsGPU[gTid];
    }
  }
}

__global__
void fourthPassInitKernel(
    const GraphDataParallelGPU graphDP1GPU,
    const GraphDataParallelGPU graphDP2GPU,
    const int* reachableGPU,
    const int* newNodesOffsetGPU,
    GraphDataParallelGPU newGraphDPGPU,
    int* toExploreGPU,
    int* newNodesVisitedGPU,
    int numNodesFirst,
    int numNodes) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numNodes) {
    int2 indices = OneDToTwoDIndexGPU(gTid, numNodesFirst);

    if (graphDP1GPU.start[indices.x] && graphDP2GPU.start[indices.y]) {
      if (reachableGPU[gTid]) {
        toExploreGPU[gTid] = true;
        newNodesVisitedGPU[gTid] = true;
        newGraphDPGPU.start[newNodesOffsetGPU[gTid]] = true;
        newGraphDPGPU.accept[newNodesOffsetGPU[gTid]] =
           graphDP1GPU.accept[indices.x] && graphDP2GPU.accept[indices.y];
      }
    }
  }
}

__global__
void secondPassInitKernel(
    const GraphDataParallelGPU graphDP1GPU,
    const GraphDataParallelGPU graphDP2GPU,
    const int* reachableGPU,
    int* toExploreGPU,
    int* newNodesGPU,
    int numNodesFirst,
    int numNodes) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numNodes) {
    int2 indices = OneDToTwoDIndexGPU(gTid, numNodesFirst);

    if (graphDP1GPU.start[indices.x] && graphDP2GPU.start[indices.y]) {
      if (reachableGPU[gTid]) {
        toExploreGPU[gTid] = true;
        newNodesGPU[gTid] = true;
      }
    }
  }
}

__global__
void findReachableInitInitKernel(
    const GraphDataParallelGPU graphDP1GPU,
    const GraphDataParallelGPU graphDP2GPU,
    int* reachableGPU,
    int* toExploreGPU,
    int numNodesFirst,
    int numNodes) {
  const int gTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gTid < numNodes) {
    int2 indices = OneDToTwoDIndexGPU(gTid, numNodesFirst);

    if (graphDP1GPU.accept[indices.x] && graphDP2GPU.accept[indices.y]) {
      toExploreGPU[gTid] = true;
      reachableGPU[gTid] = true;
    }
  }
}

} // namespace

Graph compose(const Graph& first, const Graph& second) {
  GraphDataParallel graphDP1, graphDP2;

  // Convert from AOS to SOA
  graphDP1 = convertToDataParallel(first);
  graphDP2 = convertToDataParallel(second);

  // Copy to GPU
  GraphDataParallelGPU graphDP1GPU, graphDP2GPU;
  graphDP1GPU = copyToGPU(graphDP1);
  graphDP2GPU = copyToGPU(graphDP2);
  
  const int numAllPairNodes = first.numNodes() * second.numNodes();
  const int numNodesFirst = first.numNodes();

  // Fixed number of CUDA threads and stream for all kernels
  const int NT = 128;

  //////////////////////////////////////////////////////////////////////////
  // Step 1: Data parallel findReachable
  //////////////////////////////////////////////////////////////////////////

  int* reachableGPU;
  int* epsilonMatchedGPU;
  int* toExploreGPU;

  cudaMalloc((void **)(&reachableGPU), sizeof(int) * numAllPairNodes);
  cudaMalloc((void **)(&epsilonMatchedGPU), sizeof(int) * numAllPairNodes);
  cudaMalloc((void **)(&toExploreGPU), sizeof(int) * numAllPairNodes);

  cudaMemset((void*)reachableGPU, false, sizeof(int) * numAllPairNodes);
  cudaMemset((void*)epsilonMatchedGPU, false, sizeof(int) * numAllPairNodes);
  cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes);

  {
    const int gridSize = div_up(numAllPairNodes, NT);

    findReachableInitInitKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU,
      reachableGPU, toExploreGPU, numNodesFirst, numAllPairNodes);
  }

  // std::cout << "num all pair nodes " << numAllPairNodes << std::endl;

  // This is the outer control loop that would spawn DP kernels
  while(checkAnyTrueGPU(toExploreGPU, numAllPairNodes)) {

    int* toExploreNodePairFirstGPU;
    int* toExploreNodePairSecondGPU;
    size_t numToExploreNodePair;

    // Convert bits set in toExplore to node pairs
    std::tie(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, numToExploreNodePair) =
      convertToNodePairGPU(toExploreGPU, numAllPairNodes, numNodesFirst);

    int* arcCrossProductIndexGPU;
    int* toExploreNumArcsFirstGPU;
    int* toExploreNumArcsSecondGPU;

    std::tie(arcCrossProductIndexGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU) =
      calculateArcCrossProductOffsetGPU(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU,
        numToExploreNodePair, graphDP1GPU, graphDP2GPU, true);

    int* arcCrossProductOffsetGPU;
    size_t numArcCrossProductOffset;
    int totalArcs;

    std::tie(arcCrossProductOffsetGPU, numArcCrossProductOffset, totalArcs) =
      prefixSumScanGPU(arcCrossProductIndexGPU, numToExploreNodePair, true);
    assert(numArcCrossProductOffset == (numToExploreNodePair + 1));

    cudaFree(arcCrossProductIndexGPU);

    // Reset so pristine state for next frontier to explore
    cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes);

    if (totalArcs > 0) {

      const int gridSize = div_up(totalArcs, NT);

      findReachableKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU, arcCrossProductOffsetGPU,
		      toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU, toExploreNodePairFirstGPU,
		      toExploreNodePairSecondGPU, numNodesFirst, totalArcs, numArcCrossProductOffset,
		      toExploreGPU, reachableGPU, epsilonMatchedGPU);
    }

    cudaFree(toExploreNodePairFirstGPU);
    cudaFree(toExploreNodePairSecondGPU);
    cudaFree(arcCrossProductOffsetGPU);
    cudaFree(toExploreNumArcsFirstGPU);
    cudaFree(toExploreNumArcsSecondGPU);
  } // end while for findReachable

  //////////////////////////////////////////////////////////////////////////
  // Step 2: Compute a) valid nodes in combined graph
  //                 b) Number of in and out arcs in combined graph
  // This information is used to generate offsets for nodes and arcs
  // in the combined graph
  //////////////////////////////////////////////////////////////////////////

  int* newNodesGPU;
  int* numOutArcsGPU;
  int* numInArcsGPU;

  cudaMalloc((void **)(&newNodesGPU), sizeof(int) * numAllPairNodes);
  cudaMalloc((void **)(&numOutArcsGPU), sizeof(int) * numAllPairNodes);
  cudaMalloc((void **)(&numInArcsGPU), sizeof(int) * numAllPairNodes);

  cudaMemset((void*)newNodesGPU, false, sizeof(int) * numAllPairNodes);
  cudaMemset((void*)numOutArcsGPU, 0, sizeof(int) * numAllPairNodes);
  cudaMemset((void*)numInArcsGPU, 0, sizeof(int) * numAllPairNodes);

  cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes);

  {
    const int gridSize = div_up(numAllPairNodes, NT);

    secondPassInitKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU, reachableGPU,
      toExploreGPU, newNodesGPU, numNodesFirst, numAllPairNodes);
  }

  // This is the outer control loop that would spawn DP kernels
  while(checkAnyTrueGPU(toExploreGPU, numAllPairNodes)) {

    int* toExploreNodePairFirstGPU;
    int* toExploreNodePairSecondGPU;
    size_t numToExploreNodePair;

    // Convert bits set in toExplore to node pairs
    std::tie(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, numToExploreNodePair) =
      convertToNodePairGPU(toExploreGPU, numAllPairNodes, numNodesFirst);

    int* arcCrossProductIndexGPU;
    int* toExploreNumArcsFirstGPU;
    int* toExploreNumArcsSecondGPU;

    std::tie(arcCrossProductIndexGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU) =
      calculateArcCrossProductOffsetGPU(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU,
        numToExploreNodePair, graphDP1GPU, graphDP2GPU, false);

    int* arcCrossProductOffsetGPU;
    size_t numArcCrossProductOffset;
    int totalArcs;

    std::tie(arcCrossProductOffsetGPU, numArcCrossProductOffset, totalArcs) =
      prefixSumScanGPU(arcCrossProductIndexGPU, numToExploreNodePair, true);
    assert(numArcCrossProductOffset == (numToExploreNodePair + 1));

    cudaFree(arcCrossProductIndexGPU);

    // Reset so pristine state for next frontier to explore
    cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes);

    if (totalArcs > 0) {

      const int gridSize = div_up(totalArcs, NT);

      computeValidNodeAndArcKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU,
        arcCrossProductOffsetGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU,
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, reachableGPU,
        epsilonMatchedGPU, numNodesFirst, totalArcs, numArcCrossProductOffset,
        toExploreGPU, newNodesGPU, numInArcsGPU, numOutArcsGPU);
    }

    cudaFree(toExploreNodePairFirstGPU);
    cudaFree(toExploreNodePairSecondGPU);
    cudaFree(arcCrossProductOffsetGPU);
    cudaFree(toExploreNumArcsFirstGPU);
    cudaFree(toExploreNumArcsSecondGPU);
  }

  //////////////////////////////////////////////////////////////////////////
  // Step 3: Generate offsets for nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////
  GraphDataParallelGPU newGraphDPGPU;

  int totalNodes;
  int* newNodesOffsetGPU;
  size_t numElements;
  std::tie(newNodesOffsetGPU, numElements, totalNodes) = prefixSumScanGPU(newNodesGPU, numAllPairNodes, false);
  assert(numElements == numAllPairNodes);

  newGraphDPGPU.numNodes = totalNodes;
  cudaMalloc((void **)(&(newGraphDPGPU.start)), sizeof(int) * totalNodes);
  cudaMalloc((void **)(&(newGraphDPGPU.accept)), sizeof(int) * totalNodes);
  cudaMalloc((void **)(&(newGraphDPGPU.inArcOffset)), sizeof(int) * totalNodes);
  cudaMalloc((void **)(&(newGraphDPGPU.outArcOffset)), sizeof(int) * totalNodes);

  // Generate offsets for nodes and arcs
  {
    const int NT = 128;
    const int gridSize = div_up(numAllPairNodes, NT);

    calculateNumArcsKernel<<<gridSize, NT, 0, 0>>>(newNodesGPU, newNodesOffsetGPU,
      numInArcsGPU, numOutArcsGPU, newGraphDPGPU.inArcOffset, newGraphDPGPU.outArcOffset,
      numAllPairNodes, totalNodes);
  }

  int totalInArcs;
  int totalOutArcs;

  int* inArcOffsetGPU;
  int* outArcOffsetGPU;

  std::tie(inArcOffsetGPU, numElements, totalInArcs) = prefixSumScanGPU(newGraphDPGPU.inArcOffset, totalNodes, false);
  assert(numElements == totalNodes);

  std::tie(outArcOffsetGPU, numElements, totalOutArcs) = prefixSumScanGPU(newGraphDPGPU.outArcOffset, totalNodes, false);
  assert(numElements == totalNodes);

  assert(totalInArcs == totalOutArcs);
  newGraphDPGPU.numArcs = totalOutArcs;
  cudaMalloc((void **)(&(newGraphDPGPU.inArcs)), sizeof(int) * totalInArcs);
  cudaMalloc((void **)(&(newGraphDPGPU.outArcs)), sizeof(int) * totalOutArcs);
  cudaMalloc((void **)(&(newGraphDPGPU.ilabels)), sizeof(int) * totalOutArcs);
  cudaMalloc((void **)(&(newGraphDPGPU.olabels)), sizeof(int) * totalOutArcs);
  cudaMalloc((void **)(&(newGraphDPGPU.srcNodes)), sizeof(int) * totalOutArcs);
  cudaMalloc((void **)(&(newGraphDPGPU.dstNodes)), sizeof(int) * totalOutArcs);
  cudaMalloc((void **)(&(newGraphDPGPU.weights)), sizeof(float) * totalOutArcs);

  cudaMemcpy((void *)(newGraphDPGPU.inArcOffset), (void *)(inArcOffsetGPU), sizeof(int) * totalNodes, cudaMemcpyDeviceToDevice);
  cudaMemcpy((void *)(newGraphDPGPU.outArcOffset), (void *)(outArcOffsetGPU), sizeof(int) * totalNodes, cudaMemcpyDeviceToDevice);

  // std::cout << "totalInArcs " << totalInArcs << " totalOutArcs " << totalOutArcs << std::endl;

  // SOA for gradInfo
  std::pair<std::vector<int>, std::vector<int>> gradInfo;
  gradInfo.first.resize(totalOutArcs);
  gradInfo.second.resize(totalOutArcs);

  int *gradInfoFirstGPU;
  int *gradInfoSecondGPU;

  cudaMalloc((void **)(&gradInfoFirstGPU), sizeof(int) * totalOutArcs);
  cudaMalloc((void **)(&gradInfoSecondGPU), sizeof(int) * totalOutArcs);

  //////////////////////////////////////////////////////////////////////////
  // Step 4: Generate nodes and arcs in combined graph
  //////////////////////////////////////////////////////////////////////////

  int* newNodesVisitedGPU;
  cudaMalloc((void **)(&newNodesVisitedGPU), sizeof(int) * numAllPairNodes);
  cudaMemset((void*)newNodesVisitedGPU, false, sizeof(int) * numAllPairNodes);

  // Reset so pristine state for next frontier to explore
  cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes);
  cudaMemset((void *)(newGraphDPGPU.start), false, sizeof(int) * totalNodes);
  cudaMemset((void *)(newGraphDPGPU.accept), false, sizeof(int) * totalNodes);

  {
    const int gridSize = div_up(numAllPairNodes, NT);

    fourthPassInitKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU, reachableGPU,
      newNodesOffsetGPU, newGraphDPGPU, toExploreGPU, newNodesVisitedGPU,
      numNodesFirst, numAllPairNodes);
  }

  // This is the outer control loop that would spawn DP kernels
  while(checkAnyTrueGPU(toExploreGPU, numAllPairNodes)) {

    int* toExploreNodePairFirstGPU;
    int* toExploreNodePairSecondGPU;
    size_t numToExploreNodePair;

    // Convert bits set in toExplore to node pairs
    std::tie(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, numToExploreNodePair) =
      convertToNodePairGPU(toExploreGPU, numAllPairNodes, numNodesFirst);

    int* arcCrossProductIndexGPU;
    int* toExploreNumArcsFirstGPU;
    int* toExploreNumArcsSecondGPU;

    std::tie(arcCrossProductIndexGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU) =
      calculateArcCrossProductOffsetGPU(toExploreNodePairFirstGPU, toExploreNodePairSecondGPU,
        numToExploreNodePair, graphDP1GPU, graphDP2GPU, false);

    int* arcCrossProductOffsetGPU;
    size_t numArcCrossProductOffset;
    int totalArcs;

    std::tie(arcCrossProductOffsetGPU, numArcCrossProductOffset, totalArcs) =
      prefixSumScanGPU(arcCrossProductIndexGPU, numToExploreNodePair, true);
    assert(numArcCrossProductOffset == (numToExploreNodePair + 1));

    cudaFree(arcCrossProductIndexGPU);

    // Reset so pristine state for next frontier to explore
    cudaMemset((void*)toExploreGPU, false, sizeof(int) * numAllPairNodes);

    if (totalArcs > 0) {

      const int gridSize = div_up(totalArcs, NT);

      generateNodeAndArcKernel<<<gridSize, NT, 0, 0>>>(graphDP1GPU, graphDP2GPU,
        arcCrossProductOffsetGPU, toExploreNumArcsFirstGPU, toExploreNumArcsSecondGPU,
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU, reachableGPU,
        epsilonMatchedGPU, numNodesFirst, totalArcs, numArcCrossProductOffset,
        newGraphDPGPU, toExploreGPU, gradInfoFirstGPU, gradInfoSecondGPU,
        newNodesOffsetGPU, newNodesVisitedGPU);
    }

    cudaFree(toExploreNodePairFirstGPU);
    cudaFree(toExploreNodePairSecondGPU);
    cudaFree(arcCrossProductOffsetGPU);
    cudaFree(toExploreNumArcsFirstGPU);
    cudaFree(toExploreNumArcsSecondGPU);
  }

  // Reset incremented offsets to original value
  cudaMemcpy((void *)(newGraphDPGPU.inArcOffset), (void *)(inArcOffsetGPU), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToDevice);
  cudaMemcpy((void *)(newGraphDPGPU.outArcOffset), (void *)(outArcOffsetGPU), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToDevice);

  // Copy graph on GPU to CPU
  GraphDataParallel newGraphDP;
  newGraphDP.start.resize(totalNodes);
  newGraphDP.accept.resize(totalNodes);
  newGraphDP.inArcOffset.resize(totalNodes);
  newGraphDP.outArcOffset.resize(totalNodes);
  newGraphDP.inArcs.resize(totalInArcs);
  newGraphDP.outArcs.resize(totalOutArcs);
  newGraphDP.ilabels.resize(totalOutArcs);
  newGraphDP.olabels.resize(totalOutArcs);
  newGraphDP.srcNodes.resize(totalOutArcs);
  newGraphDP.dstNodes.resize(totalOutArcs);
  newGraphDP.weights.resize(totalOutArcs);
 
  cudaMemcpy((void *)(newGraphDP.accept.data()), (void *)(newGraphDPGPU.accept), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)(newGraphDP.start.data()), (void *)(newGraphDPGPU.start), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)(newGraphDP.inArcOffset.data()), (void *)(newGraphDPGPU.inArcOffset), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)(newGraphDP.outArcOffset.data()), (void *)(newGraphDPGPU.outArcOffset), sizeof(int) * newGraphDPGPU.numNodes, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)(newGraphDP.inArcs.data()), (void *)(newGraphDPGPU.inArcs), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)(newGraphDP.outArcs.data()), (void *)(newGraphDPGPU.outArcs), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)(newGraphDP.ilabels.data()), (void *)(newGraphDPGPU.ilabels), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)(newGraphDP.olabels.data()), (void *)(newGraphDPGPU.olabels), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)(newGraphDP.srcNodes.data()), (void *)(newGraphDPGPU.srcNodes), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)(newGraphDP.dstNodes.data()), (void *)(newGraphDPGPU.dstNodes), sizeof(int) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)(newGraphDP.weights.data()), (void *)(newGraphDPGPU.weights), sizeof(float) * newGraphDPGPU.numArcs, cudaMemcpyDeviceToHost);

  assert(newGraphDPGPU.numArcs == totalOutArcs);
  cudaMemcpy((void *)(gradInfo.first.data()), (void *)(gradInfoFirstGPU), sizeof(int) * totalOutArcs, cudaMemcpyDeviceToHost);
  cudaMemcpy((void *)(gradInfo.second.data()), (void *)(gradInfoSecondGPU), sizeof(int) * totalOutArcs, cudaMemcpyDeviceToHost);

  cudaFree(reachableGPU);
  cudaFree(epsilonMatchedGPU);
  cudaFree(toExploreGPU);
  cudaFree(newNodesGPU);
  cudaFree(numOutArcsGPU);
  cudaFree(numInArcsGPU);
  cudaFree(newNodesOffsetGPU);
  cudaFree(inArcOffsetGPU);
  cudaFree(outArcOffsetGPU);
  cudaFree(gradInfoFirstGPU);
  cudaFree(gradInfoSecondGPU);
  cudaFree(newNodesVisitedGPU);

  cudaFree(newGraphDPGPU.start);
  cudaFree(newGraphDPGPU.accept);
  cudaFree(newGraphDPGPU.inArcOffset);
  cudaFree(newGraphDPGPU.outArcOffset);
  cudaFree(newGraphDPGPU.inArcs);
  cudaFree(newGraphDPGPU.outArcs);
  cudaFree(newGraphDPGPU.ilabels);
  cudaFree(newGraphDPGPU.olabels);
  cudaFree(newGraphDPGPU.srcNodes);
  cudaFree(newGraphDPGPU.dstNodes);
  cudaFree(newGraphDPGPU.weights);
  newGraphDPGPU.numNodes = 0;
  newGraphDPGPU.numArcs = 0;

  if (0)
  {
    std::cout << "nodes " << newGraphDP.inArcOffset.size() << std::endl;
    std::cout << "nodes " << newGraphDP.outArcOffset.size() << std::endl;

    std::cout << "start" << std::endl;
    for (auto i : newGraphDP.start) {
      std::cout << i << std::endl;
    }

    std::cout << "accept" << std::endl;
    for (auto i : newGraphDP.accept) {
      std::cout << i << std::endl;
    }

    std::cout << "inArcOffset" << std::endl;
    for (auto i : newGraphDP.inArcOffset) {
      std::cout << i << std::endl;
    }

    std::cout << "outArcOffset" << std::endl;
    for (auto i : newGraphDP.outArcOffset) {
      std::cout << i << std::endl;
    }

    std::cout << "inArcs" << std::endl;
    for (auto i : newGraphDP.inArcs) {
      std::cout << i << std::endl;
    }

    std::cout << "outArcs" << std::endl;
    for (auto i : newGraphDP.outArcs) {
      std::cout << i << std::endl;
    }

    std::cout << "ilabels" << std::endl;
    for (auto i : newGraphDP.ilabels) {
      std::cout << i << std::endl;
    }

    std::cout << "olabels" << std::endl;
    for (auto i : newGraphDP.olabels) {
      std::cout << i << std::endl;
    }

    std::cout << "srcNodes" << std::endl;
    for (auto i : newGraphDP.srcNodes) {
      std::cout << i << std::endl;
    }

    std::cout << "dstNodes" << std::endl;
    for (auto i : newGraphDP.dstNodes) {
      std::cout << i << std::endl;
    }

    std::cout << "weights" << std::endl;
    for (auto i : newGraphDP.weights) {
      std::cout << i << std::endl;
    }
  }
  // Not needed since the CPU data is never incremented
  // Shift offset values back down after adding arcs to newGraphDP
  // The offset values got converted from exclusive prefix sum to inclusive
  // Need to convert them back to exclusive prefix sum  by starting with 0
  // and shifting to right by 1
  // for (int i = newGraphDP.outArcOffset.size() - 1; i >= 0; --i) {
    // newGraphDP.outArcOffset[i] = i == 0 ? 0 : newGraphDP.outArcOffset[i - 1];
    // newGraphDP.inArcOffset[i] = i == 0 ? 0 : newGraphDP.inArcOffset[i - 1];
  // }

  // Convert back and add in autograd metadata
  auto nGraph = convertFromDataParallel(newGraphDP);
  nGraph.setInputs({first, second});

  if (0)
  {
    std::cout << "numNodes " << nGraph.numNodes() << std::endl;

    std::cout << "accept" << std::endl;
    for (auto i : nGraph.accept()) {
      std::cout << i << std::endl;
    }

    std::cout << "start" << std::endl;
    for (auto i : nGraph.start()) {
      std::cout << i << std::endl;
    }

    std::cout << "numIn" << std::endl;
    for (int i = 0; i < nGraph.numNodes(); ++i) {
      std::cout << nGraph.numIn(i) << std::endl;
    }

    std::cout << "numOut" << std::endl;
    for (int i = 0; i < nGraph.numNodes(); ++i) {
      std::cout << nGraph.numOut(i) << std::endl;
    }
  }

  // Convert gradInfo SOA to AOS
  std::vector<std::pair<int, int>> gradInfoAOS;
  for (int i = 0; i < gradInfo.first.size(); ++i) {
    gradInfoAOS.emplace_back(gradInfo.first[i], gradInfo.second[i]);
  }

  // TODO eliminate this copy pasta.
  auto gradFunc = [gradInfo = std::move(gradInfoAOS)](
                      std::vector<Graph>& inputs, Graph deltas) {
    // In this case the arc's parents are always from the
    // first and second input graphs respectively.
    bool calcGrad1 = inputs[0].calcGrad();
    bool calcGrad2 = inputs[1].calcGrad();
    auto grad1 = calcGrad1 ? std::vector<float>(inputs[0].numArcs(), 0.0)
                           : std::vector<float>{};
    auto grad2 = calcGrad2 ? std::vector<float>(inputs[1].numArcs(), 0.0)
                           : std::vector<float>{};
    for (int i = 0; i < gradInfo.size(); i++) {
      auto arcGrad = deltas.weight(i);
      auto& arcs = gradInfo[i];
      if (calcGrad1 && arcs.first >= 0) {
        grad1[arcs.first] += arcGrad;
      }
      if (calcGrad2 && arcs.second >= 0) {
        grad2[arcs.second] += arcGrad;
      }
    }
    inputs[0].addGrad(std::move(grad1));
    inputs[1].addGrad(std::move(grad2));
  };
  nGraph.setGradFunc(std::move(gradFunc));
  return nGraph;
}

} // namespace dataparallel
} // namespace detail
} // namespace gtn

    /*
    if (0)
    {
      int *aCPGPU;
      int *tEN1GPU;
      int *tEN2GPU;

      std::tie(aCPGPU, tEN1GPU, tEN2GPU) = calculateArcCrossProductOffsetGPU(
        toExploreNodePairFirstGPU, toExploreNodePairSecondGPU,
        toExploreNodePair.first.size(), graphDP1GPU, graphDP2GPU, true);

      std::vector<int> aCP(numToExploreNodePair);
      std::vector<int> tEN1(numToExploreNodePair);
      std::vector<int> tEN2(numToExploreNodePair);

      cudaMemcpy((void *)(aCP.data()), (void *)(aCPGPU), sizeof(int) * numToExploreNodePair, cudaMemcpyDeviceToHost);
      cudaMemcpy((void *)(tEN1.data()), (void *)(tEN1GPU), sizeof(int) * numToExploreNodePair, cudaMemcpyDeviceToHost);
      cudaMemcpy((void *)(tEN2.data()), (void *)(tEN2GPU), sizeof(int) * numToExploreNodePair, cudaMemcpyDeviceToHost);

      assert(std::equal(arcCrossProductOffset.begin(), arcCrossProductOffset.end(), aCP.begin()));
      assert(std::equal(toExploreNumArcs.first.begin(), toExploreNumArcs.first.end(), tEN1.begin()));
      assert(std::equal(toExploreNumArcs.second.begin(), toExploreNumArcs.second.end(), tEN2.begin()));

      cudaFree(aCPGPU);
      cudaFree(tEN1GPU);
      cudaFree(tEN2GPU);
    }*/

    /*
    if(0)
    {
      std::vector<int> tVec(arcCrossProductOffset);
      const size_t numElts = tVec.size();
      int* tVecGPU;
      cudaMalloc((void **)(&tVecGPU), sizeof(int) * numElts);
      cudaMemcpy((void *)tVecGPU, (void *)(tVec.data()), sizeof(int) * numElts, cudaMemcpyHostToDevice);

      const int totalArcs = prefixSumScan(tVec, true);
      int* tVecScanGPU;
      size_t tVecScanElts;
      int tArcsGPU;
      std::tie(tVecScanGPU, tVecScanElts, tArcsGPU) = prefixSumScanGPU(tVecGPU, numElts, true);

      assert(tVec.size() == (numElts + 1));
      assert(tVecScanElts == (numElts + 1));
      std::vector<int> tVecNew(tVec.size());
      cudaMemcpy((void *)(tVecNew.data()), (void *)(tVecScanGPU), sizeof(int) * tVecScanElts, cudaMemcpyDeviceToHost);

      assert(totalArcs == tArcsGPU);
      assert(std::equal(tVec.begin(), tVec.end(), tVecNew.begin()));

      cudaFree(tVecGPU);
      cudaFree(tVecScanGPU);
    }*/

    /*
    if (0)
    {
      int* tEN1GPU;
      int* tEN2GPU;
      size_t nTEN;

      std::tie(tEN1GPU, tEN2GPU, nTEN) = convertToNodePairGPU(toExploreGPU, numAllPairNodes, numNodesFirst);

      assert(nTEN == toExploreNodePair.first.size());

      std::vector<int> tEN1(nTEN);
      std::vector<int> tEN2(nTEN);
      cudaMemcpy((void *)(tEN1.data()), (void *)(tEN1GPU), sizeof(int) * nTEN, cudaMemcpyDeviceToHost);
      cudaMemcpy((void *)(tEN2.data()), (void *)(tEN2GPU), sizeof(int) * nTEN, cudaMemcpyDeviceToHost);

      assert(std::equal(toExploreNodePair.first.begin(), toExploreNodePair.first.end(), tEN1.begin()));
      assert(std::equal(toExploreNodePair.second.begin(), toExploreNodePair.second.end(), tEN2.begin()));

      cudaFree(tEN1GPU);
      cudaFree(tEN2GPU);
    }*/

/*
inline std::pair<int, int> OneDToTwoDIndex(int n, int n1Extent) {
  assert(n1Extent > 0);
  const int n2 = n / n1Extent;
  const int n1 = n % n1Extent;
  return std::make_pair(n1, n2);
}


bool checkAnyTrue(const std::vector<int>& flags) {
  // Potentially wasteful - but GPU friendly
  return std::accumulate(flags.begin(), flags.end(), 0) > 0 ? true : false;
}*/


/*
// Convert int array to pairs for true flags
std::pair<std::vector<int>, std::vector<int>> convertToNodePair(
    const std::vector<int>& flags,
    int extent) {
  std::vector<int> indices(flags);
  const int numValidNodes = prefixSumScan(indices, false);

  std::vector<int> toExploreNodePairFirst(numValidNodes);
  std::vector<int> toExploreNodePairSecond(numValidNodes);

  // No loop dependence
  for (size_t i = 0; i < flags.size(); ++i) {
    if (flags[i] == true) {
      std::pair<int, int> node = OneDToTwoDIndex(i, extent);

      const int index = indices[i];
      assert(index >= 0);
      assert(index < numValidNodes);
      toExploreNodePairFirst[index] = node.first;
      toExploreNodePairSecond[index] = node.second;
    }
  }

  return std::make_pair(toExploreNodePairFirst, toExploreNodePairSecond);
}*/

// Takes a pair of nodes, where each member of pair comes from a different
// graph and calculate a vector of number of arcs in the cross product of
// arcs outgoing from each pair.
// This should be a kernel call
/*
std::tuple<std::vector<int>, std::pair<std::vector<int>, std::vector<int>>>
calculateArcCrossProductOffset(
    const std::pair<std::vector<int>, std::vector<int>>& toExploreNodePair,
    const GraphDataParallel& graphDP1,
    const GraphDataParallel& graphDP2,
    bool inOrOutArc) {
  assert(toExploreNodePair.first.size() == toExploreNodePair.second.size());

  std::pair<std::vector<int>, std::vector<int>> toExploreNumArcs;
  toExploreNumArcs.first.resize(toExploreNodePair.first.size());
  toExploreNumArcs.second.resize(toExploreNodePair.first.size());

  std::vector<int> arcCrossProductOffset(toExploreNodePair.first.size());

  // No dependence between iterations
  for (size_t i = 0; i < toExploreNodePair.first.size(); ++i) {
    int node = (toExploreNodePair.first)[i];
    // Special case if it is the last node. Then the offset becomes
    // the number of arcs
    const int inArcOffsetGraph1 = ((node + 1) == graphDP1.inArcOffset.size())
        ? graphDP1.inArcs.size()
        : graphDP1.inArcOffset[node + 1];
    const int outArcOffsetGraph1 = ((node + 1) == graphDP1.outArcOffset.size())
        ? graphDP1.outArcs.size()
        : graphDP1.outArcOffset[node + 1];

    const int numArcsFirst = inOrOutArc
        ? inArcOffsetGraph1 - graphDP1.inArcOffset[node]
        : outArcOffsetGraph1 - graphDP1.outArcOffset[node];

    node = (toExploreNodePair.second)[i];
    // Special case if it is the last node. Then the offset becomes
    // the number of arcs
    const int inArcOffsetGraph2 = ((node + 1) == graphDP2.inArcOffset.size())
        ? graphDP2.inArcs.size()
        : graphDP2.inArcOffset[node + 1];
    const int outArcOffsetGraph2 = ((node + 1) == graphDP2.outArcOffset.size())
        ? graphDP2.outArcs.size()
        : graphDP2.outArcOffset[node + 1];

    const int numArcsSecond = inOrOutArc
        ? inArcOffsetGraph2 - graphDP2.inArcOffset[node]
        : outArcOffsetGraph2 - graphDP2.outArcOffset[node];

    (toExploreNumArcs.first)[i] = numArcsFirst;
    (toExploreNumArcs.second)[i] = numArcsSecond;

    // Even when numArcsFirst or numArcsSecond is 0 we have to consider
    // the case when the other graph has arcs with epsilon label
    if (numArcsFirst != 0 && numArcsSecond != 0) {
      arcCrossProductOffset[i] = numArcsFirst * numArcsSecond;
    } else if (numArcsFirst != 0 && numArcsSecond == 0) {
      arcCrossProductOffset[i] = numArcsFirst;
    } else if (numArcsFirst == 0 && numArcsSecond != 0) {
      arcCrossProductOffset[i] = numArcsSecond;
    } else {
      arcCrossProductOffset[i] = 0;
    }
  }

  return std::make_tuple(arcCrossProductOffset, toExploreNumArcs);
}*/
