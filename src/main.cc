/*
 * @Author: zhubin
 * @Date: 2023-02-27 17:44:25
 * @FilePath: \leetcode\src\main.cc
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name}, All Rights Reserved.
 */

#include "leetcode.h"

#include <cstdio>
#include <memory>
#include <string>

int main(int argc, char const *argv[]) {
    std::unique_ptr<leetcode::Solution> solution = std::make_unique<leetcode::Solution>();
    // std::string a = "abc";
    // std::string b = "bc";
    // int c = solution->minDistance(a, b);
    // printf("%d\n", c);
    string s = "aab";

    vector<vector<string>> o = solution->str_partition(s);

    return 0;
}