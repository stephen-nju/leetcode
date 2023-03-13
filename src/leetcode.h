#ifndef LEET_CODE_H

#define LEET_CODE_H
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using std::string;
using std::vector;
namespace leetcode {

struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution
{
  public:
    int minDistance(string word1, string word2);
    // 10. 正则表达式匹配
    bool isMatch(string s, string p);
    // 二叉树最大直径
    int diameterOfBinaryTree(TreeNode *root);

    void quickSort(int arr[], int length);

    // 蓝桥杯2022 ，c组整数拆分(二维背包问题)
    int64_t intPartition();

    // 121.买卖股票的最佳时机
    int maxProfit(vector<int> &prices);
    // 122.买卖股票的最佳时机 II
    int maxProfit_2(vector<int> &prices);
    // 123. 买卖股票的最佳时机 III
    int maxProfit_3(vector<int> &prices);

    // 77.组合
    vector<vector<int>> combine(int n, int k);

    // 216. 组合总和 III
    vector<vector<int>> combinationSum3(int k, int n);

    // 17. 电话号码的字母组合
    vector<string> letterCombinations(string digits);

    // 39. 组合总和
    vector<vector<int>> combinationSum(vector<int> &candidates, int target);
    // 40. 组合总和 II
    vector<vector<int>> combinationSum2(vector<int> &candidates, int target);
    // 131. 分割回文串
    vector<vector<string>> str_partition(string s);
    // 93. 复原 IP 地址
    vector<string> restoreIpAddresses(string s);
    // 78. 子集
    vector<vector<int>> subsets(vector<int> &nums);
    // 90. 子集 II
    vector<vector<int>> subsetsWithDup(vector<int> &nums);


  private:
    void quickSort(int arr[], int left, int right);
};

}// namespace leetcode

#endif// leetcode_h