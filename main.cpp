/**
 * @file main.cpp
 *
 * @breif Experimentation with B-Trees
 *
 * @date 1/05/2018
 * @author Ben Caine
 */

#include "external/cpp-btree/btree_map.h"

int main() {
    btree::btree_map<int, std::string> map;

    map.insert({5, "hello"});

    return 0;
}