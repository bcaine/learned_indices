/**
 * @file DataUtils.h
 *
 * @breif Data utilities to help with the Recursive Model Index
 *
 * @date 1/07/2018
 * @author Ben Caine
 */

#ifndef LEARNED_INDICES_DATAUTILS_H
#define LEARNED_INDICES_DATAUTILS_H

#include <unordered_set>
#include <random>
#include <chrono>
#include <cassert>

/**
 * @brief Get a random batch to train on
 * @tparam KeyType [in]: The key type of our data
 * @param batchSize [in]: How many elements we want
 * @param datasetSize [in]: The total dataset size
 * @return A set of indices in the data that we want to use in this batch
 */
template <typename KeyType>
std::unordered_set<KeyType> getRandomBatch(int batchSize, int datasetSize) {
    assert(datasetSize > batchSize && "Dataset size is smaller than requested batch size, which causes an infinite loop");

    std::unordered_set<KeyType> randomKeys;
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<KeyType> distribution(0, datasetSize);

    while (randomKeys.size() < batchSize) {
        auto val = distribution(rng);
        randomKeys.insert(val);
    }
    return randomKeys;
}

#endif //LEARNED_INDICES_DATAUTILS_H
