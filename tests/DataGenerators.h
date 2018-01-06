/**
 * @file DataGenerators.h
 *
 * @breif Data generation functions for testing B-Trees and Learned Indices
 *
 * @date 1/04/2018
 * @author Ben Caine
 */

#ifndef AUTOINDEX_DATAGENERATORS_H
#define AUTOINDEX_DATAGENERATORS_H

#include <random>

/**
 * @brief Generate integer type lognormals scaled to a desired max value
 * @tparam Dtype [in]: An integer type (int, long, size_t)
 * @tparam Length [in]: Length of the dataset
 * @param desiredMaxValue [in]: Desired max value to scale last value to
 * @param mean [in]: Mean of our lognormal distribution
 * @param stddev [in]: Standard deviation of our log normal distribution
 * @return An array of sorted values of type Dtype and length Length, with the max value being desiredMaxValue
 */
template <typename Dtype, int Length>
std::array<Dtype, Length> getIntegerLognormals(double desiredMaxValue = 1e7, double mean = 0.0, double stddev = 2.0) {
    std::default_random_engine generator;
    std::lognormal_distribution<double> distribution(mean, stddev);

    std::array<double, Length> doubleValues;
    for (size_t ii = 0; ii < Length; ++ii) {
        doubleValues[ii] = distribution(generator);
    }

    std::sort(doubleValues.begin(), doubleValues.end());

    double maxValue = doubleValues[doubleValues.size() - 1];
    double scalingFactor = desiredMaxValue / maxValue;

    std::array<Dtype, Length> returnValues;
    for (size_t ii = 0; ii < doubleValues.size(); ++ii) {
        returnValues[ii] = static_cast<Dtype>(doubleValues[ii] * scalingFactor);
    }

    return returnValues;
};


#endif //AUTOINDEX_DATAGENERATORS_H
