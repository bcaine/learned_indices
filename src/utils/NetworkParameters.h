/**
 * @file NetworkParameters.h
 *
 * @breif A simple way to store network params
 *
 * @date 1/07/2018
 * @author Ben Caine
 */
#ifndef LEARNED_INDICES_NETWORKPARAMETERS_H
#define LEARNED_INDICES_NETWORKPARAMETERS_H

/**
 * @brief A container for the hyperparameters of our first level network
 */
struct NetworkParameters {
    int batchSize;      ///< The batch size of our network
    int maxNumEpochs;   ///< The max number of epochs to train the network for
    float learningRate; ///< The learning rate of our Adam solver
    int numNeurons;     ///< The number of neurons
};

#endif //LEARNED_INDICES_NETWORKPARAMETERS_H
