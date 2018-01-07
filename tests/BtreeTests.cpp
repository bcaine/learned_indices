/**
 * @file BtreeTests.cpp
 *
 * @breif Experimentation with B-Trees
 *
 * @date 1/04/2018
 * @author Ben Caine
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE BtreeTests

#include <boost/test/unit_test.hpp>
#include <chrono>
#include "../external/cpp-btree/btree_map.h"
#include "../src/utils/DataGenerators.h"

BOOST_AUTO_TEST_CASE(btree_basic_test) {
    btree::btree_map<int, std::string> map;

    int totalInserts = 10000;
    for (int ii = 0; ii < totalInserts; ++ii) {
        map.insert({ii, "Hello"});
    }

    int queryKey = 5000;
    auto result = map.find(queryKey);
    BOOST_ASSERT_MSG(result != map.end(), "Did not return a point");
    BOOST_ASSERT_MSG(result->first == queryKey, "Result did not match query key");
    BOOST_ASSERT_MSG(result->second == "Hello", "Result value did not match");
}

BOOST_AUTO_TEST_CASE(btree_lognormal_test) {
    const size_t length = 10000;
    auto values = getIntegerLognormals<size_t, length>();

    btree::btree_map<size_t, std::string> map;
    for (const auto &value : values) {
        map.insert({value, "Hello"});
    }

    const int sampleSize = 100;
    std::mt19937 rng;
    std::uniform_int_distribution<size_t> gen(0, values.size()); // uniform, unbiased

    std::cout << values.size() << std::endl;
    std::array<size_t, sampleSize> queryKeys;
    for (int ii = 0; ii < sampleSize; ++ii) {
        queryKeys[ii] = gen(rng);
    }

    std::array<double, sampleSize> durations;

    for (int ii = 0; ii < sampleSize; ++ii) {
        size_t key = values[queryKeys[ii]];
        auto startTime = std::chrono::system_clock::now();
        auto result = map.find(key);
        auto endTime = std::chrono::system_clock::now();

        std::chrono::duration<double> duration = endTime - startTime;
        durations[ii] = duration.count();
    }

    double average = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
    double min = *std::min(durations.begin(), durations.end());
    double max = *std::max(durations.begin(), durations.end());

    std::cout << "Min: " << min << std::endl;
    std::cout << "Average: " << average << std::endl;
    std::cout << "Max: " << max << std::endl;
}