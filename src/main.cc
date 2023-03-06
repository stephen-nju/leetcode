
#include "leetcode.h"
#include <cstdio>
#include <memory>
#include <string>

int main(int argc, char const *argv[])
{
    std::unique_ptr<leetcode::Solution> solution = std::make_unique<leetcode::Solution>();
    std::string a = "abc";
    std::string b = "bc";
    int c = solution->minDistance(a, b);
    printf("%d\n", c);
    return 0;
}