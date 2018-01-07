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
#include <iomanip>
#include <fstream>
#include <unordered_set>
#include "DataGenerators.h"
#include "../external/nn_cpp/nn/Net.h"


std::unordered_set<size_t> getRandomSubset(int numValues, int maxValue) {
    std::unordered_set<size_t> values;
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

    // Hyperparams
    int batchSize = 128;
    int numNeurons = 8;
    float learningRate = 0.005;
    int numEpochs = 25000;
    const size_t datasetSize = 500000;
    float maxValue = 1e5;

    auto values = getIntegerLognormals<size_t, datasetSize>(maxValue);

    // File to write loss data to
    std::ofstream outputFile("loss.csv");

    bool useBias = true;
    // Simple linear model
    nn::Net<float> net;
    net.add(new nn::Dense<float, 2>(batchSize, 1, numNeurons, useBias, nn::InitializationScheme::GlorotNormal));
    net.add(new nn::Relu<float, 2>());
    net.add(new nn::Dense<float, 2>(batchSize, numNeurons, 1, useBias, nn::InitializationScheme::GlorotNormal));


    // NOTE: MSE blows up the gradients. Smooth L1 Loss behaves much nicer here
    nn::HuberLoss<float, 2> lossFunction;

    // NOTE: SGD simply does not work. Adam works in Pytorch, with a lot of sensitivity to learning rate
    net.registerOptimizer(new nn::Adam<float>(learningRate));

    Eigen::Tensor<float, 2> input(batchSize, 1);
    Eigen::Tensor<float, 2> positions(batchSize, 1);

    auto startTime = std::chrono::system_clock::now();
    for (unsigned int currentEpoch = 0; currentEpoch < numEpochs; ++currentEpoch) {
        auto newBatch = getRandomSubset(batchSize, datasetSize);
        unsigned int ii = 0;
        for (auto idx : newBatch) {
            input(ii, 0) = static_cast<float>(values[idx]);
            positions(ii, 0) = static_cast<float>(idx);
            ii ++;
        }

        auto result = net.forward<2, 2>(input);
        result = result * result.constant(datasetSize);

        auto loss = lossFunction.loss(result, positions);
        std::cout << "Epoch: " << currentEpoch << " Loss: " << loss << std::endl;
        outputFile << currentEpoch << ", " << loss << "\n";

        net.backward<2>(lossFunction.backward(result, positions));
        net.step();
    }
    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<float> duration = endTime - startTime;
    std::cout << "Total training of " << numEpochs << " iters took: " << duration.count() << "s" << std::endl;

    outputFile.close();

    // Test
    auto newBatch = getRandomSubset(batchSize, datasetSize);
    unsigned int ii = 0;
    for (auto idx : newBatch) {
        input(ii, 0) = static_cast<float>(values[idx]);
        positions(ii, 0) = static_cast<float>(idx);
        ii ++;
    }
    auto result = net.forward<2, 2>(input);

    for (unsigned int row = 0; row < batchSize; ++row) {
        std::cout << std::fixed << std::setprecision(0) << input(row, 0) << ", " << positions(row, 0) << ", " << result(row, 0) * datasetSize << std::endl;
    }
}