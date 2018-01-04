cmake_minimum_required(VERSION 3.0)
project(autoindex CXX)
#include_directories(nn)

set(CMAKE_CXX_STANDARD 11)

# Add nn_cpp
add_subdirectory(external/nn_cpp)

file(GLOB cpp_btree_sources external/cpp-btree/*.h)
add_library(cpp_btree STATIC ${cpp_btree_sources})
target_include_directories(cpp_btree PUBLIC cpp_btree)
set_target_properties(cpp_btree PROPERTIES LINKER_LANGUAGE CXX)

add_executable(autoindex main.cpp)
target_link_libraries(autoindex cpp_btree neural_net)