/**
 * @file RecursiveModelIndex.h
 *
 * @breif An implementation of the Recursive Model Index concept
 *
 * @date 1/07/2018
 * @author Ben Caine
 */

#ifndef LEARNED_INDICES_RECURSIVEMODELINDEX_H
#define LEARNED_INDICES_RECURSIVEMODELINDEX_H

#include "SecondStageNode.h"
#include "utils/DataUtils.h"
#include "utils/NetworkParameters.h"
#include "../external/nn_cpp/nn/Net.h"
#include "../external/cpp-btree/btree_map.h"
#include <boost/optional.hpp>


/**
 * @brief An implementation of the recursive model index
 * @tparam KeyType: The key type of our index
 * @tparam ValueType: The value we are storing
 * @tparam secondStageSize: The size of our second stage of our index
 */
template <typename KeyType, typename ValueType, int secondStageSize>
class RecursiveModelIndex {
public:

    /**
     * @brief Create a RMI
     * @param firstStageParams [in]: The first layer network parameters
     * @param secondStageParams [in]: The second stage network parameters
     * @param maxSecondStageError [in]: The max second stage error allowed before replacing with BTree
     * @param maxOverflowSize [in]: The max size our overflow BTree can get to before we force a retrain
     */
    explicit RecursiveModelIndex(const NetworkParameters &firstStageParams,
                                 const NetworkParameters &secondStageParams,
                                 int maxSecondStageError = 256,
                                 int maxOverflowSize = 10000);

    //TODO: Is it more common to pass a pair?
    /**
     * @brief Insert into our index new data
     * @param key [in]: The key to insert
     * @param value [in]: The value to insert
     */
    void insert(KeyType key, ValueType value);

    //TODO: What to do if not found
    /**
     * @brief Find a specific item from the tree
     * @param key [in]: A key to search for
     * @return A pair of (key, value) if found.
     */
    boost::optional<std::pair<KeyType, ValueType>> find(KeyType key);

    /**
     * @brief Train our index structure
     */
    void train();

private:

    /**
     * @brief Train the first stage of the network
     */
    void trainFirstStage();

    /**
     * @brief train the second stage linear models of the network
     */
    void trainSecondStage();

    ///------------ Data members ----------------
    std::vector<std::pair<KeyType, ValueType>> m_data;                 ///< The data our learned index tries to find

    NetworkParameters m_firstStageParams;                              ///< First stage network parameters
    NetworkParameters m_secondStageParams;                             ///< Our second stage network parameters
    std::unique_ptr<nn::Net<float>> m_firstStageNetwork;               ///< The first stage neural network
    std::vector<SecondStageNode<KeyType>> m_secondStage;                   ///< The second stage (network or btree)
    int m_maxSecondStageError;                                         ///< Max second stage error before replacing with btree

    int m_currentOverflowSize;                                         ///< Number of inserts stored in overflow array
    int m_maxOverflowSize;                                             ///< Max size we let overflow array get before retraining
    std::vector<std::pair<KeyType, ValueType>> m_overflowArray;        ///< The overflow array
};


template <typename KeyType, typename ValueType, int secondStageSize>
RecursiveModelIndex<KeyType, ValueType, secondStageSize>::RecursiveModelIndex(const NetworkParameters &firstStageParams,
                                                                              const NetworkParameters &secondStageParams,
                                                                              int maxSecondStageError,
                                                                              int maxOverflowSize):
    m_firstStageParams(firstStageParams), m_secondStageParams(secondStageParams),
    m_maxSecondStageError(maxSecondStageError), m_maxOverflowSize(maxOverflowSize)
{

    // Create our first network
    m_firstStageNetwork.reset(new nn::Net<float>());
    m_firstStageNetwork->add(new nn::Dense<float, 2>(firstStageParams.batchSize, 1, firstStageParams.numNeurons, true, nn::InitializationScheme::GlorotNormal));
    m_firstStageNetwork->add(new nn::Relu<float, 2>());
    m_firstStageNetwork->add(new nn::Dense<float, 2>(firstStageParams.batchSize, firstStageParams.numNeurons, 1, true, nn::InitializationScheme::GlorotNormal));

    // Create all our second stage models
    for (size_t ii = 0; ii < secondStageSize; ++ii) {
        m_secondStage.emplace_back(SecondStageNode<KeyType>(m_maxSecondStageError, secondStageParams.batchSize));
    }
}

template <typename KeyType, typename ValueType, int secondStageSize>
void RecursiveModelIndex<KeyType, ValueType, secondStageSize>::insert(KeyType key, ValueType value) {
    m_overflowArray.push_back({key, value});
    m_currentOverflowSize ++;

    // TODO: This should really be a background task
    if (m_currentOverflowSize > m_maxOverflowSize) {
        train();
    }
};

template <typename KeyType, typename ValueType, int secondStageSize>
boost::optional<std::pair<KeyType, ValueType>> RecursiveModelIndex<KeyType, ValueType, secondStageSize>::find(KeyType key) {
    // TODO: Order of searching?
    auto overflowResult = std::find_if(m_overflowArray.begin(), m_overflowArray.end(), [&](const std::pair<KeyType, ValueType> &pair) {
        return pair.first == key;
    });

    if (overflowResult != m_overflowArray.end()) {
        return *overflowResult;
    }

    // Now search using the RecursiveModelIndex!
    Eigen::Tensor<float, 2> input(1, 1);
    input(0, 0) = static_cast<float>(key);

    auto result = m_firstStageNetwork->forward<2, 2>(input);
    auto resultIdx = result * result.constant(m_data.size());

    // Calculate which stage we want to send this data to
    // If we take the result (unscaled, so closer to 0-1), and multiply by the
    // number of stages we get an assignment
    int stage = static_cast<int>(result(0, 0) * secondStageSize);

    // Cap the range of stages to 0 -> (secondStageSize - 1)
    stage = std::max(0, stage);
    stage = std::min(secondStageSize - 1, stage);

    std::cout << "Finding: " << key << " Predicted: " << resultIdx << " assigned to stage: " << stage << std::endl;

    if (m_secondStage[stage].isValid()) {
        if (m_secondStage[stage].useTree()) {

            std::cout << "Using tree" << std::endl;
            auto treeResult = m_secondStage[stage].treeFind(key);
            if (treeResult) {
                return {key, m_data[treeResult.get().second]};
            } else {
                return {};
            }
        } else {
            // TODO: Too much casting, long vs size_t vs int... Clean this mess up. Bugs have to be everywhere
            long predictedIdx = m_secondStage[stage].predict(key, m_data.size());
            // Search from min to max around predictedIdx
            size_t startIdx = std::max(static_cast<long>(0), predictedIdx + m_secondStage[stage].getMaxNegativeError());
            size_t endIdx = std::min(m_data.size() - 1, static_cast<size_t>(predictedIdx + m_secondStage[stage].getMaxPositiveError()));

            auto findResult = std::find_if(m_data.begin() + startIdx, m_data.begin() + endIdx,
                                           [&](const std::pair<KeyType, ValueType> &pair) {
                                               return pair.first == key;
                                           });

            if (findResult != m_data.begin() + endIdx) {
                return *findResult;
            } else {
                return {};
            }
        }
    } else {
        std::cerr << "Key: " << key << " requested an invalid stage two node" << std::endl;
    }

    return {};
};

template <typename KeyType, typename ValueType, int secondStageSize>
void RecursiveModelIndex<KeyType, ValueType, secondStageSize>::train() {
    std::cout << "Retraining..." << std::endl;
    m_data.insert(m_data.end(), m_overflowArray.begin(), m_overflowArray.end());

    // Sort data
    std::sort(m_data.begin(), m_data.end(), [](std::pair<KeyType, ValueType> p1, std::pair<KeyType, ValueType> p2) {
        return p1.first < p2.first;
    });

    trainFirstStage();
    trainSecondStage();

    // Clear out overflow tree
    m_overflowArray.clear();
    m_currentOverflowSize = 0;
}

template <typename KeyType, typename ValueType, int secondStageSize>
void RecursiveModelIndex<KeyType, ValueType, secondStageSize>::trainFirstStage() {
    // TODO: Do we want to clear out the old network or use it's previous weights?
    std::cout << "Training first stage" << std::endl;

    // Huber loss is used for increased stability
    nn::HuberLoss<float, 2> lossFunction;

    // Adam because vanilla SGD doesn't converge at all
    m_firstStageNetwork->registerOptimizer(new nn::Adam<float>(m_firstStageParams.learningRate));

    Eigen::Tensor<float, 2> input(m_firstStageParams.batchSize, 1);
    Eigen::Tensor<float, 2> positions(m_firstStageParams.batchSize, 1);

    for (int currentEpoch = 0; currentEpoch < m_firstStageParams.maxNumEpochs; ++currentEpoch) {
        auto newBatch = getRandomBatch<KeyType>(m_firstStageParams.batchSize, m_data.size());
        int ii = 0;
        for (auto idx : newBatch) {
            // Input is the key
            input(ii, 0) = static_cast<float>(m_data[idx].first);
            // Label is the position in our sorted array
            positions(ii, 0) = static_cast<float>(idx);
            ii++;
        }

        auto result = m_firstStageNetwork->forward<2, 2>(input);
        result = result * result.constant(m_data.size());

        auto loss = lossFunction.loss(result, positions);
        // TODO: Add logging, make this Debug
        std::cout << "Epoch: " << currentEpoch << " Loss: " << loss << std::endl;

        auto lossBack = lossFunction.backward(result, positions);
        // Divide loss back by dataset size to stabilize training and remove relationship between
        // learning rate and dataset size
        lossBack = lossBack / lossBack.constant(m_data.size());

        m_firstStageNetwork->backward<2>(lossBack);
        m_firstStageNetwork->step();
    }
}

template <typename KeyType, typename ValueType, int secondStageSize>
void RecursiveModelIndex<KeyType, ValueType, secondStageSize>::trainSecondStage() {
    std::cout << "Creating per stage dataset" << std::endl;

    // Create training sets for second stage models
    std::array<std::vector<std::pair<KeyType, size_t>>, secondStageSize> perStageDataset;
    Eigen::Tensor<float, 2> predictInput(1, 1);
    for (int ii = 0; ii < m_data.size(); ++ii) {
        predictInput(0, 0) = static_cast<float>(m_data[ii].first);

        // Get result from first stage, and then scale
        auto result = m_firstStageNetwork->forward<2, 2>(predictInput);
        auto resultIdx = result * result.constant(m_data.size());

        // Calculate which stage we want to send this data to
        // If we take the result (unscaled, so closer to 0-1), and multiply by the
        // number of stages we get an assignment
        int stage = static_cast<int>(result(0, 0) * secondStageSize);

        // Cap the range of stages to 0 -> (secondStageSize - 1)
        stage = std::max(0, stage);
        stage = std::min(secondStageSize - 1, stage);
        perStageDataset[stage].push_back({m_data[ii].first, ii});
    }

    std::cout << "Training second stage" << std::endl;
    // Train each stage
    for (int stage = 0; stage < secondStageSize; ++stage) {
        m_secondStage[stage].train(perStageDataset[stage], m_secondStageParams, m_data.size());
    }
}

#endif //LEARNED_INDICES_RECURSIVEMODELINDEX_H
