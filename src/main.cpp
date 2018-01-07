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
    NetworkParameters params;
    params.batchSize = 256;
    params.maxNumEpochs = 25000;
    params.learningRate = 0.01;
    params.numNeurons = 8;

    RecursiveModelIndex<int, int, 100> recursiveModelIndex(params, 100000);

    const size_t datasetSize = 100010;
    float maxValue = 1e5;

    auto values = getIntegerLognormals<int, datasetSize>(maxValue);
    for (auto val : values) {
        recursiveModelIndex.insert(val, val + 1);
    }


    auto result = recursiveModelIndex.find(5000);

    if (result.first == 0) {
        std::cout << "Could not find" << std::endl;
    } else {
        std::cout << result.first << ", " << result.second << std::endl;
    }

    return 0;
}