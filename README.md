## A C++11 implementation of the B-Tree part of "The Case for Learned Index Structures"

A research **proof of concept** that implements the B-Tree section of [The Case for Learned Index Structures](https://arxiv.org/pdf/1712.01208.pdf) paper in C++.

The general design is to have a single lookup structure that you can parameterize with a KeyType and a ValueType, and an overflow list that keeps new inserts until you retrain. There is a value in the constructor of the RMI that triggers a retrain when the overflow array reaches a certain size.

The basic API:

```c++
auto modelIndex = RecursiveModelIndex<int, int, 128> recursiveModelIndex(firstStageParams, secondStageParams, 256, 1e6);
 
modelIndex.insert(5, 15);
modelIndex.insert(15, 5);
 
auto result = modelIndex.find(5);
 
if (result) {
    std::cout << "Yay! We got: " << result.get().first << ", " << result.get().second << std::endl;
} else {
    std::cout << "We didn't find your value." << std::endl; // This shouldn't happen in the above usage...
}
```

See [src/main.cpp](src/main.cpp) for a usage example where it stores scaled log normal data.

### Dependencies

- [nn_cpp](https://github.com/bcaine/nn_cpp) - Eigen based minimalistic C++ Neural Network library
- [cpp-btree](https://code.google.com/archive/p/cpp-btree/) - A fast C++ implementation of a B+ Tree

### TODO:

- Lots of code cleanup
- Profiling of where the slowdowns are. On small tests, the cpp_btree lib beats it by 10-100x
    - Eigen::TensorFixed in nn_cpp would definitely help
    - Increasing dataset size may lead to more of an advantage to the RMI
    - Being much, much more efficient with memory and conversions (lots of casting)
- Very non-linear data the second stage tends to break down or stop performing on. 
- A non-trivial amount of our second stage "dies" in the sense that we don't use it for predictions. 
    - The larger the dataset, or the more second stage nodes, the more likely this is. Bug somewhere?
- Experimenting/tuning of training parameters
    - Still more learning rate sensitive than I'd like
- Checking, and failing if there are non-integer keys
- Tests on the actual RMI code (instead of using tests for experiments)
- Move retrain to non-blocking thread
- Logging


