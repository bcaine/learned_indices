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
    firstStageParams.learningRate = 0.01;
    firstStageParams.numNeurons = 8;

    NetworkParameters secondStageParams;
    secondStageParams.batchSize = 64;
    secondStageParams.maxNumEpochs = 1000;
    secondStageParams.learningRate = 0.01;

    RecursiveModelIndex<int, int, 100> recursiveModelIndex(firstStageParams, secondStageParams, 256, 1e6);

    const size_t datasetSize = 10000;
    float maxValue = 1e5;

    auto values = getIntegerLognormals<int, datasetSize>(maxValue);
    for (auto val : values) {
        recursiveModelIndex.insert(val, val + 1);
    }

    recursiveModelIndex.train();

    for (unsigned int ii = 0; ii < datasetSize; ii += 500) {
        auto result = recursiveModelIndex.find(values[ii]);
        if (result) {
            std::cout << result.get().first << ", " << result.get().second << std::endl;
        } else {
            std::cout << "Failed to find value that should be there" << std::endl;
        }
    }

    return 0;
}