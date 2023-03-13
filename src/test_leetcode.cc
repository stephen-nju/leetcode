#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "leetcode.h"

#include <string>
using std::string;

TEST_CASE("testing minDistance") {
    string a                     = "a";
    string b                     = "abc";
    leetcode::Solution *solution = new leetcode::Solution();
    CHECK(solution->minDistance(a, b) == 3);
}