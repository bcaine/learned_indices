/**
 * @file main.cpp
 *
 * @breif An implementation of the B-Tree part of the learned indices paper
 *
 * @date 1/05/2018
 * @author Ben Caine
 */

#include "utils/DataGenerators.h"
#include "RecursiveModelIndex.h"

int main() {
    NetworkParameters firstStageParams;
    firstStageParams.batchSize = 256;
    firstStageParams.maxNumEpochs = 25000;
    firstStageParams.learningRate = 0.005;
    firstStageParams.numNeurons = 8;

    NetworkParameters secondStageParams;
    secondStageParams.batchSize = 64;
    secondStageParams.maxNumEpochs = 1000;
    secondStageParams.learningRate = 0.01;

    RecursiveModelIndex<int, int, 100> recursiveModelIndex(firstStageParams, secondStageParams, 10000000);

    const size_t datasetSize = 10000;
    float maxValue = 1e5;

    auto values = getIntegerLognormals<int, datasetSize>(maxValue);
    for (auto val : values) {
        recursiveModelIndex.insert(val, val + 1);
    }

    recursiveModelIndex.train();

    auto result = recursiveModelIndex.find(5000);

    if (result.first == 0) {
        std::cout << "Could not find" << std::endl;
    } else {
        std::cout << result.first << ", " << result.second << std::endl;
    }

    return 0;
}