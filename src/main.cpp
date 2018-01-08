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
#include <algorithm>

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

    RecursiveModelIndex<int, int, 128> recursiveModelIndex(firstStageParams, secondStageParams, 256, 1e6);
    btree::btree_map<int, int> btreeMap;

    const size_t datasetSize = 10000;
    float maxValue = 1e4;

    auto values = getIntegerLognormals<int, datasetSize>(maxValue);
    for (auto val : values) {
        recursiveModelIndex.insert(val, val + 1);
        btreeMap.insert({val, val + 1});
    }

    recursiveModelIndex.train();

    std::vector<double> rmiDurations;
    std::vector<double> btreeDurations;
    for (unsigned int ii = 0; ii < datasetSize; ii += 500) {

        auto startTime = std::chrono::system_clock::now();
        auto result = recursiveModelIndex.find(values[ii]);
        auto endTime = std::chrono::system_clock::now();

        std::chrono::duration<double> duration = endTime - startTime;
        rmiDurations.push_back(duration.count());

        if (result) {
            std::cout << result.get().first << ", " << result.get().second << std::endl;
        } else {
            std::cout << "Failed to find value that should be there" << std::endl;
        }

        startTime = std::chrono::system_clock::now();
        auto btreeResult = btreeMap.find(values[ii]);
        endTime = std::chrono::system_clock::now();
        duration = endTime - startTime;
        btreeDurations.push_back(duration.count());
    }

    auto summaryStats = [](const std::vector<double> &durations) {
        double average = std::accumulate(durations.cbegin(), durations.cend() - 1, 0.0) / durations.size();
        auto minmax = std::minmax(durations.cbegin(), durations.cend() - 1);

        std::cout << "Min: " << *minmax.first << std::endl;
        std::cout << "Average: " << average << std::endl;
        std::cout << "Max: " << *minmax.second << std::endl;
    };

    std::cout << std::endl << std::endl;
    std::cout << "Recursive Model Index Timings" << std::endl;
    summaryStats(rmiDurations);

    std::cout << std::endl << std::endl;
    std::cout << "BTree Timings" << std::endl;
    summaryStats(btreeDurations);

    return 0;
}