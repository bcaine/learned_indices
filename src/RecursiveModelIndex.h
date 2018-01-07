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

#include "../external/nn_cpp/nn/Net.h"
#include "../external/cpp-btree/btree_map.h"

/**
 * @brief A container for the hyperparameters of our first level network
 */
struct NetworkParameters {
    int batchSize;      ///< The batch size of our network
    int maxNumEpochs;   ///< The max number of epochs to train the network for
    float learningRate; ///< The learning rate of our Adam solver
    int numNeurons;     ///< The number of neurons
};

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
     * @param networkParameters [in]: The first layer network parameters
     * @param maxOverflowSize [in]: The max size our overflow BTree can get to before we force a retrain
     */
    explicit RecursiveModelIndex(const NetworkParameters &networkParameters, int maxOverflowSize = 10000);

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
    std::pair<KeyType, ValueType> find(KeyType key);

    /**
     * @brief Train our index structure
     */
    void train();

private:
    std::unique_ptr<nn::Net<float>> m_firstStageNetwork;               ///< The first stage neural network
    std::array<nn::Net<float>, secondStageSize> m_secondStageNetworks; ///< The second stage networks
    int m_maxOverflowSize;                                             ///< Max size we let overflow tree get before retraining
    btree::btree_map<KeyType, ValueType> m_overflowTree;               ///< A tree to catch all overflows
};

template <typename KeyType, typename ValueType, int secondStageSize>
RecursiveModelIndex<KeyType, ValueType, secondStageSize>::RecursiveModelIndex(const NetworkParameters &networkParameters, int maxOverflowSize):
    m_maxOverflowSize(maxOverflowSize)
{

    // Create our first network
    m_firstStageNetwork.reset(new nn::Net<float>());
    m_firstStageNetwork->add(new nn::Dense<float, 2>(networkParameters.batchSize, 1, networkParameters.numNeurons, true, nn::InitializationScheme::GlorotNormal));
    m_firstStageNetwork->add(new nn::Relu<float, 2>());
    m_firstStageNetwork->add(new nn::Dense<float, 2>(networkParameters.batchSize, networkParameters.numNeurons, 1, true, nn::InitializationScheme::GlorotNormal));

    // Create all our linear models
    for (size_t ii = 0; ii < secondStageSize; ++ii) {
        m_secondStageNetworks[ii] = nn::Net<float>();
        m_secondStageNetworks[ii].add(new nn::Dense<float, 2>(networkParameters.batchSize, 1, 1, true, nn::InitializationScheme::GlorotNormal));
    }
}

template <typename KeyType, typename ValueType, int secondStageSize>
void RecursiveModelIndex<KeyType, ValueType, secondStageSize>::insert(KeyType key, ValueType value) {
    m_overflowTree.insert({key, value});

    // TODO: Force retrain here?

};

template <typename KeyType, typename ValueType, int secondStageSize>
std::pair<KeyType, ValueType> RecursiveModelIndex<KeyType, ValueType, secondStageSize>::find(KeyType key) {
    // TODO: Order of searching?
    auto result = m_overflowTree.find(key);
    if (result != m_overflowTree.end()) {
        return *result;
    }

    return {};
};

#endif //LEARNED_INDICES_RECURSIVEMODELINDEX_H
