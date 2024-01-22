/*
 * @Author: zhubin
 * @Date: 2023-02-27 14:37:37
 * @FilePath: \leetcode\src\leetcode.h
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name}, All Rights Reserved.
 */
#ifndef LEET_CODE_H

#define LEET_CODE_H
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdint.h>
#include <string>
#include <vector>

using std::string;
using std::vector;

typedef uint8_t nlp_uint8_t;
typedef int8_t nlp_int8_t;
typedef int64_t nlp_int64_t;

namespace leetcode {

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right)
        : val(x), left(left), right(right) {}
};

// N叉树的Node
class Node {
  public:
    int val;
    vector<Node *> children;
    Node() {}
    Node(int _val) { val = _val; }
    Node(int _val, vector<Node *> _children) {
        val = _val;
        children = _children;
    }
};

class Solution {
  public:
    // ===============动态规划========================
    int minDistance(string word1, string word2);
    // 10. 正则表达式匹配
    bool isMatch(string s, string p);
    // 二叉树最大直径
    int diameterOfBinaryTree(TreeNode *root);

    void quickSort(int arr[], int length);

    // 蓝桥杯2022 ，c组整数拆分(二维背包问题)
    int64_t intPartition();

    // 647. 回文子串
    int countSubstrings(string s);

    // 509 斐波那契数列
    int fib(int n);
    // 70 爬楼梯
    int climbStairs(int n);
    // 746 最小代价爬楼梯
    int minCostClimbingStairs(vector<int> &cost);
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
    // 491. 递增子序列
    vector<vector<int>> findSubsequences(vector<int> &nums);
    // 46. 全排列
    vector<vector<int>> permute(vector<int> &nums);
    //     47. 全排列 II
    vector<vector<int>> permuteUnique(vector<int> &nums);
    // 332. 重新安排行程(Hierholzer 算法)
    vector<string> findItinerary(vector<vector<string>> &tickets);
    // 51. N 皇后
    vector<vector<string>> solveNQueens(int n);
    // 37. 解数独
    void solveSudoku(vector<vector<char>> &board);
    // ========================================二叉树遍历==========================
    // 144. 二叉树的前序遍历
    vector<int> preorderTraversal(TreeNode *root);
    // 145.⼆叉树的后序遍历
    vector<int> postorderTraversal(TreeNode *root);

    // 94. 二叉树的中序遍历
    vector<int> inorderTraversal(TreeNode *root);

    // 102. 二叉树的层序遍历
    vector<vector<int>> levelOrder(TreeNode *root);
    // 107. 二叉树的层序遍历 II
    vector<vector<int>> levelOrderBottom(TreeNode *root);
    // 199. 二叉树的右视图
    vector<int> rightSideView(TreeNode *root);
    // 637. 二叉树的层平均值
    vector<double> averageOfLevels(TreeNode *root);

    // 226. 翻转二叉树
    TreeNode *invertTree(TreeNode *root);

    // =============二叉树判别===========
    // 101. 对称二叉树
    bool isSymmetric(TreeNode *root);

    // 98. 验证二叉搜索树
    bool isValidBST(TreeNode *root);
    // 104. 二叉树的最大深度
    int maxDepth(TreeNode *root);
    // 559. N 叉树的最大深度
    int maxDepth_N(Node *root);
    // 111. 二叉树的最小深度
    int minDepth(TreeNode *root);
    // 110. 平衡二叉树
    bool isBalanced(TreeNode *root);
    // 257.二叉树所有路径
    vector<string> binaryTreePaths(TreeNode *root);
    // 404. 左叶子之和
    int sumOfLeftLeaves(TreeNode *root);
    // 513. 找树左下角的值
    int findBottomLeftValue(TreeNode *root);
    // 106. 从中序与后序遍历序列构造二叉树
    TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder);
    // 617.合并二叉树
    TreeNode *mergeTrees(TreeNode *root1, TreeNode *root2);
    // 700.二叉搜索树中的搜索
    TreeNode *searchBST(TreeNode *root, int val);

    // 236. 二叉树的最近公共祖先
    TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q);

    // 448.
    vector<int> findDisappearedNumbers(vector<int> &nums);

    //===============栈=============
    // 1047. 删除字符串中的所有相邻重复项
    string removeDuplicates(string s);

    //============单调栈============
    // 739.每日温度
    vector<int> dailyTemperatures(vector<int> &temperatures);

    //==============堆=============
    // 692. 前K个高频单词
    vector<string> topKFrequent(vector<string> &words, int k);
    // 23. 合并 K 个升序链表
    ListNode *mergeKLists(vector<ListNode *> &lists);
    // 662. 二叉树最大宽度
    int widthOfBinaryTree(TreeNode *root);

    //===============贪心算法==============================

    // 455. 分发饼干
    int findContentChildren(vector<int> &g, vector<int> &s);

  private:
    void quickSort(int arr[], int left, int right);
};

} // namespace leetcode

#endif // leetcode_h
