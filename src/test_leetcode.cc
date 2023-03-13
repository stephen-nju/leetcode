#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "leetcode.h"

#include <string>
using std::string;

leetcode::Solution *solution = new leetcode::Solution();
TEST_CASE("testing minDistance") {
    string a = "a";
    string b = "abc";
    CHECK(solution->minDistance(a, b) == 2);
}

TEST_CASE("testing findSubsequences") {
    vector<int> nums{ 4, 6, 7, 7 };
    vector<vector<int>> ground_truth = {
        { { 4, 6 }, { 4, 7 }, { 4, 6, 7 }, { 4, 6, 7, 7 }, { 6, 7 }, { 6, 7, 7 }, { 7, 7 }, { 4, 7, 7 } }
    };
    vector<vector<int>> o = solution->findSubsequences(nums);
    for (auto s : o) {
        for (auto ss : s) { printf("%d", ss); }
    }
}

TEST_CASE("testing permute") {
    vector<int> nums{ 1, 2, 3 };
    vector<vector<int>> o = solution->permute(nums);
    
}