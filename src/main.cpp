/**
 * @file main.cpp
 *
 * @breif An implementation of the B-Tree part of the learned indices paper
 *
 * @date 1/05/2018
 * @author Ben Caine
 */

#include "RecursiveModelIndex.h"

int main() {
    NetworkParameters params;
    params.batchSize = 128;
    params.maxNumEpochs = 10000;
    params.learningRate = 0.01;
    params.numNeurons = 8;

    RecursiveModelIndex<int, int, 100> recursiveModelIndex(params);


    int numValues = 10000;
    for (int ii = 0; ii < numValues; ++ii) {
        recursiveModelIndex.insert(ii, ii * 2);
    }

    auto result = recursiveModelIndex.find(5000);

    if (result.first == 0) {
        std::cout << "Could not find" << std::endl;
    } else {
        std::cout << result.first << ", " << result.second << std::endl;
    }

    return 0;
}