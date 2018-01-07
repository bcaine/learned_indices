cmake_minimum_required(VERSION 3.0)
project(learned_indices CXX)

option(LEARNED_INDICES_BUILD_TESTS "Whether to build tests" ON)
set(CMAKE_CXX_STANDARD 11)

# Add nn_cpp
add_subdirectory(external/nn_cpp)

file(GLOB cpp_btree_sources external/cpp-btree/*.h)
add_library(cpp_btree STATIC ${cpp_btree_sources})
target_include_directories(cpp_btree PUBLIC cpp_btree)
set_target_properties(cpp_btree PROPERTIES LINKER_LANGUAGE CXX)

add_executable(learned_indices main.cpp)
target_link_libraries(learned_indices cpp_btree nn_cpp)

if (LEARNED_INDICES_BUILD_TESTS)
    find_package(Boost COMPONENTS unit_test_framework)

    if (Boost_FOUND)
        enable_testing()
        add_executable(btree_test tests/BtreeTests.cpp)
        target_link_libraries(btree_test ${Boost_LIBRARIES} cpp_btree)
        add_test(NAME btree_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND btree_test)

        add_executable(nn_index_test tests/NetworkIndexTests.cpp)
        target_link_libraries(nn_index_test ${Boost_LIBRARIES} nn_cpp)
        add_test(NAME nn_index_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND nn_index_test)
    endif()
endif()
