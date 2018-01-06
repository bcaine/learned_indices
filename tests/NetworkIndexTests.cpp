/**
 * @file BtreeTests.cpp
 *
 * @breif Experimentation with a Neural Network querying position
 *
 * @date 1/04/2018
 * @author Ben Caine
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE NetworkIndexTests

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <random>
#include <fstream>
#include <set>
#include "DataGenerators.h"
#include "../external/nn_cpp/nn/Net.h"
#include "../external/nn_cpp/nn/loss/HuberLoss.h"


std::set<size_t> getRandomSubset(int numValues, int maxValue) {
    std::set<size_t> values;
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> gen(0, maxValue);

    while (values.size() < numValues) {
        auto val = gen(rng);
        values.insert(val);
    }
    return values;
}

BOOST_AUTO_TEST_CASE(basic_net) {
    const size_t length = 100000;
    auto values = getIntegerLognormals<size_t, length>(1e6);

    // File to write loss data to
    std::ofstream outputFile("loss.csv");

    int batchSize = 64;
    int numNeurons = 32;

    // Simple linear model
    nn::Net<double> net;
    net.add(new nn::Dense<double, 2>(batchSize, 1, numNeurons, true));
    net.add(new nn::Relu<double, 2>());
    net.add(new nn::Dense<double, 2>(batchSize, numNeurons, 1, true));

    nn::HuberLoss<double, 2> lossFunction;

    int numEpochs = 100;
    float learningRate = 0.009;
    Eigen::Tensor<double, 2> input(batchSize, 1);
    Eigen::Tensor<double, 2> positions(batchSize, 1);

    for (unsigned int currentEpoch = 0; currentEpoch < numEpochs; ++currentEpoch) {
        auto newBatch = getRandomSubset(batchSize, length);
        unsigned int ii = 0;
        for (auto idx : newBatch) {
            input(ii, 0) = static_cast<double>(values[idx]);
            positions(ii, 0) = static_cast<double>(idx);
            ii ++;
        }

        auto result = net.forward<2, 2>(input);
        result = result * result.constant(length);

        auto loss = lossFunction.loss(result, positions);
        std::cout << "Epoch: " << currentEpoch << " Loss: " << loss << std::endl;
        outputFile << currentEpoch << ", " << loss << "\n";

        auto lossBack = lossFunction.backward(result, positions);
//
//        std::cout << result << std::endl;
//        std::cout << positions << std::endl;
//        std::cout << lossBack << std::endl;
        net.backward<2>(lossBack);
        net.updateWeights(learningRate);
    }

    outputFile.close();

    // Test
    auto newBatch = getRandomSubset(batchSize, length);
    unsigned int ii = 0;
    for (auto idx : newBatch) {
        input(ii, 0) = static_cast<double>(values[idx]);
        positions(ii, 0) = static_cast<double>(idx);
        ii ++;
    }

    auto result = net.forward<2, 2>(input);
    std::cout << result * result.constant(length) << std::endl;
    std::cout << positions << std::endl;

}