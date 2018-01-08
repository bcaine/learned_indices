/**
 * @file SecondStage.h
 *
 * @breif An implementation of the second stage of a recursive index
 *
 * @date 1/07/2018
 * @author Ben Caine
 */

#ifndef LEARNED_INDICES_SECONDSTAGE_H
#define LEARNED_INDICES_SECONDSTAGE_H

#include "../external/nn_cpp/nn/Net.h"
#include "../external/cpp-btree/btree_map.h"
#include "utils/DataUtils.h"
#include "utils/NetworkParameters.h"
#include <boost/optional.hpp>

// TODO: This doesn't protect against calling tree related funcs if no tree
// TODO: Nor does it protect against calling the network before training
// TODO: Make this much smarter
/**
 * @brief A wrapper around the second stage (either network or btree)
 * @tparam KeyType [in]: Keytype of the stage
 */
template <typename KeyType>
class SecondStageNode {
public:

    /**
     * @brief Create a second stage
     * @param positionErrorThreshold [in]: The error threshold before we switch to a BTree
     * @param netBatchSize [in]: The batch size to initialize the net with for training
     */
    explicit SecondStageNode(int positionErrorThreshold, int netBatchSize);

    /**
     * @brief Whether the current node is valid
     */
    bool isValid() {
        return m_nodeIsValid;
    }

    /**
     * @return Return the max negative error of this stage
     */
    int getMaxNegativeError() {
        return m_maxNegativeError;
    }

    /**
     * @return Return the max positive error of this stage
     */
    int getMaxPositiveError() {
        return m_maxPositiveError;
    }

    /**
     * @brief Predict a location with the network
     * @param key [in]: Key to use as input
     * @param totalDatasetSize [in]: The dataset size of the WHOLE dataset
     * @return A predicted location
     */
    size_t predict(KeyType key, size_t totalDatasetSize);

    /**
     * @brief Train this stages network
     * @param data [in]: A reference to the training data (key, idx)
     * @param trainingParameters [in]: The current network parameters
     * @param totalDatasetSize [in]: The size of the WHOLE dataset
     */
    void train(const std::vector<std::pair<KeyType, size_t>> &data, const NetworkParameters &trainingParameters, size_t totalDatasetSize);

    /**
     * @return Whether to use the tree
     */
    bool useTree() {
        return m_useTree;
    }

    /**
     * @brief Use the tree to find an item
     * @param key [in]: The key to use to search
     * @return A pair of key, idx if saved
     */
    boost::optional<std::pair<KeyType, size_t>> treeFind(KeyType key);

private:
    bool m_useTree;                           ///< Whether to use the tree or not
    int m_positionErrorThreshold;             ///< The max position error before swapping to a BTree
    bool m_nodeIsValid;                       ///< Whether this node is valid (has data)

    /// Net related items
    std::unique_ptr<nn::Net<float>> m_net;    ///< Our network for this stage
    int m_maxNegativeError;                   ///< Max error (negative) of a prediction
    int m_maxPositiveError;                   ///< Max error (positive) of a prediction

    /// Tree related items
    btree::btree_map<KeyType, size_t> m_tree; ///< The tree if needed
};

template <typename KeyType>
SecondStageNode<KeyType>::SecondStageNode(int positionErrorThreshold, int netBatchSize):
    m_useTree(false), m_positionErrorThreshold(positionErrorThreshold), m_nodeIsValid(false),
    m_maxNegativeError(0), m_maxPositiveError(0)
{
    // Init net
    m_net.reset(new nn::Net<float>());
    m_net->add(new nn::Dense<float, 2>(netBatchSize, 1, 1, true, nn::InitializationScheme::GlorotNormal));
}

template <typename KeyType>
boost::optional<std::pair<KeyType, size_t>> SecondStageNode<KeyType>::treeFind(KeyType key) {
    assert(m_useTree && "Called treeFind but the tree isn't supposed to be used");
    auto result = m_tree.find(key);
    if (result != m_tree.end()) {
        return std::pair<KeyType, size_t>(result->first, result->second);
    } else {
        return {};
    }
}

template <typename KeyType>
size_t SecondStageNode<KeyType>::predict(KeyType key, size_t totalDatasetSize) {
    Eigen::Tensor<float, 2> input(1, 1);
    input(0, 0) = static_cast<float>(key);

    auto result = m_net->forward<2, 2>(input);
    result = result * result.constant(totalDatasetSize);
    return static_cast<size_t>(result(0, 0));
}

template <typename KeyType>
void SecondStageNode<KeyType>::train(const std::vector<std::pair<KeyType, size_t>> &data,
                                 const NetworkParameters &trainingParameters, size_t totalDatasetSize) {
    size_t trainingDatasetSize = data.size();

    if (trainingDatasetSize == 0) {
        // TODO: Flag the object somehow...
        std::cerr << "Dataset for this stage is empty" << std::endl;
        m_nodeIsValid = false;
        return;
    }
    // If we have data, we have a valid node
    m_nodeIsValid = true;

    // Make sure batchSize is <= dataset size
    int batchSize = std::min(trainingParameters.batchSize, static_cast<int>(trainingDatasetSize));

    // If batch size is smaller than what we preassigned, reassign net
    if (batchSize < trainingParameters.batchSize) {
        m_net.reset(new nn::Net<float>());
        m_net->add(new nn::Dense<float, 2>(batchSize, 1, 1, true, nn::InitializationScheme::GlorotNormal));
    }

    m_net->registerOptimizer(new nn::Adam<float>(trainingParameters.learningRate));
    
    Eigen::Tensor<float, 2> input(batchSize, 1);
    Eigen::Tensor<float, 2> positions(batchSize, 1);
    nn::HuberLoss<float, 2> lossFunc;

    // Train this stage
    for (int currentEpoch = 0; currentEpoch < trainingParameters.maxNumEpochs; ++currentEpoch) {
        auto newBatch = getRandomBatch<KeyType>(batchSize, trainingDatasetSize);
        int ii = 0;
        for (auto idx : newBatch) {
            // In this stage, perStageDataset is pair {key, idx}
            // Input is the key
            input(ii, 0) = static_cast<float>(data[idx].first);
            // Label is the position in our sorted array
            positions(ii, 0) = static_cast<float>(data[idx].second);
            ii++;
        }

        auto result = m_net->forward<2, 2>(input);
        result = result * result.constant(totalDatasetSize);

        auto loss = lossFunc.loss(result, positions);
        auto lossBack = lossFunc.backward(result, positions);

        // TODO: Add logging, make debug message
//        std::cout << "Epoch: " << currentEpoch << " loss: " << loss << std::endl;
        lossBack = lossBack / lossBack.constant(totalDatasetSize);

        m_net->backward<2>(lossBack);
        m_net->step();
    }

    // Now calculate our error
    Eigen::Tensor<float, 2> testInput(1, 1);
    long currentMaxAbsoluteError = 0;
    m_maxNegativeError = 0;
    m_maxPositiveError = 0;

    for (int ii = 0; ii < trainingDatasetSize; ++ii) {
        const KeyType &key = data[ii].first;
        const size_t &idx = data[ii].second;
        testInput(0, 0) = static_cast<float>(key);

        auto result = m_net->forward<2, 2>(testInput);
        result = result * result.constant(totalDatasetSize);

        long predictedIdx = static_cast<long>(result(0, 0));
        auto error = static_cast<long>(idx) - predictedIdx;

        if (error < m_maxNegativeError) {
            m_maxNegativeError = error;
        }
        if (error > m_maxPositiveError) {
            m_maxPositiveError = error;
        }

        auto absError = std::abs(error);
        if (absError > currentMaxAbsoluteError) {
            currentMaxAbsoluteError = absError;
        }
    }

    if (currentMaxAbsoluteError > m_positionErrorThreshold) {
        m_useTree = true;
        for (size_t ii = 0; ii < data.size(); ++ii) {
            m_tree.insert(data[ii]);
        }
    } else {
        m_useTree = false;
    }

    std::cout << "Absolute max error: " << currentMaxAbsoluteError;
    std::cout << " Max Negative: " << m_maxNegativeError;
    std::cout << " Max Positive: " << m_maxPositiveError << std::endl;
}

#endif //LEARNED_INDICES_SECONDSTAGE_H
