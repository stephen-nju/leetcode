
#include "leetcode.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <queue>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using std::pair;
namespace leetcode {

int Solution::minDistance(string word1, string word2) {
    int distance = 0;
    int m        = word1.size();
    int n        = word2.size();
    int **dp     = (int **)malloc(sizeof(int *) * (m + 1));
    for (int i = 0; i < m + 1; i++) { dp[i] = (int *)malloc(sizeof(int *) * (n + 1)); }
    // vector<vector<int>> dp;
    for (int i = 0; i <= m; i++) { dp[i][0] = i; }
    for (int j = 0; j <= n; j++) { dp[0][j] = j; }
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i - 1] == word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = std::min(dp[i - 1][j] + 1, std::min(dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1));
            }
        }
    }
    distance = dp[m][n];
    for (int i = 0; i < m + 1; i++) { free(dp[i]); }
    free(dp);
    return distance;
}

bool Solution::isMatch(string s, string p) {
    int m = s.size();
    int n = p.size();
    // bool dp[m][n];
    vector<vector<bool>> dp;
    dp[0][0] = true;
    // 初始化数组
    for (int i = 1; i < m + 1; i++) { dp[i][0] = false; }
    for (int j = 1; j < n + 1; j++) {
        if (p[j - 1] == '*') {
            dp[0][j] = dp[0][j - 2];
        } else {
            dp[0][j] = false;
        }
    }

    for (int i = 1; i < m + 1; i++) {
        for (int j = 1; j < n + 1; j++) {
            if (s[i - 1] == p[j - 1] || p[j - 1] == '.') {
                // 先考虑第i个字符和第j个字符是否匹配
                dp[i][j] = dp[i - 1][j - 1];
            } else if (p[j - 1] == '*') {
                // 如果不匹配，现在考虑第j个字符为通配符*的情况
                if (s[i - 1] == p[j - 2] || p[j - 2] == '.') {
                    // 如果能够匹配上，可以分为匹配0次，匹配一次，匹配两次以上，如果次数大于两次以上，可以考虑去掉s末尾字符串，
                    // 保留原模式
                    dp[i][j] = dp[i][j - 2] || dp[i - 1][j - 2] || dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i][j - 2];
                }
            } else {
                dp[i][j] = false;
            }
        }
    }

    return dp[m][n];
}

int Solution::diameterOfBinaryTree(TreeNode *root) {
    std::function<int(TreeNode *, int *)> max_depth = [&](TreeNode *root, int *diameter) {
        if (root == nullptr) return 0;

        int left_depth  = max_depth(root->left, diameter);
        int right_depth = max_depth(root->right, diameter);
        *diameter       = std::max(right_depth + left_depth, *diameter);
        return std::max(left_depth, right_depth) + 1;
    };

    int diameter = 0;
    if (root == nullptr) { return 0; }
    max_depth(root, &diameter);
    return diameter;
}

// 快排
inline void swap(int arr[], int x, int y) {
    int temp;
    temp   = arr[x];
    arr[x] = arr[y];
    arr[y] = temp;
}

int partition(int arr[], int left, int right) {
    // 随机选择pivot
    int pivot_index = left + rand() % (right - left + 1);
    int pivot       = arr[pivot_index];
    swap(arr, pivot_index, right);
    int low = 0, high = right - 1;
    while (low <= high) {
        if (arr[low] <= pivot) {
            low++;
        } else if (arr[high] > pivot) {
            high--;
        } else {
            swap(arr, low++, high--);
        }
        /* code */
    }
    // low指针和high指针重合后,再进行一次比较,此时交换low和right
    swap(arr, low, right);
    return low;
}

void Solution::quickSort(int arr[], int left, int right) {
    if (left >= right) { return; }
    int pivot_index = partition(arr, left, right);
    Solution::quickSort(arr, left, pivot_index - 1);
    Solution::quickSort(arr, pivot_index + 1, right);
}

void Solution::quickSort(int arr[], int length) {
    if (length <= 1) { return; };
    Solution::quickSort(arr, 0, length - 1);
}

// 二维背包问题

int64_t Solution::intPartition() {
    const int N = 10;
    const int M = 100;
    // 创建动态数组dp[M+1][N+1][M+1]
    int64_t ***dp;
    dp = new int64_t **[M + 1];
    for (int i = 0; i < M + 1; i++) { dp[i] = new int64_t *[N + 1]; }
    for (int i = 0; i < M + 1; i++) {
        for (int j = 0; j < N + 1; j++) { dp[i][j] = new int64_t[M + 1]; }
    }

    // 数组初始化
    for (int i = 0; i < M + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            for (int k = 0; k < M + 1; k++) {
                // Assign values to the
                // memory blocks created
                dp[i][j][k] = 0;
            }
        }
    }
    // 动态规划
    int64_t ans = 0;
    dp[0][0][0] = 1;
    for (int i = 1; i <= M; i++) {
        for (int j = 0; j <= N; j++) {
            for (int k = 0; k <= M; k++) {
                if (k >= i && j >= 1) {
                    // 第i个数字小于总和

                    dp[i][j][k] = dp[i - 1][j][k] + dp[i - 1][j - 1][k - i];
                } else {
                    // 第i个数字大于总和

                    dp[i][j][k] = dp[i - 1][j][k];
                }
            }
        }
    }

    ans = dp[M][N][M];
    // Deallocate memory
    for (int i = 0; i < M + 1; i++) {
        for (int j = 0; j < N + 1; j++) { delete[] dp[i][j]; }
        delete[] dp[i];
    }
    delete[] dp;

    // ***************空间优化方案**********************///
    // dp[i][j][k] i个数总和为j并且第i位数字为k的方案数目
    // int64_t dp[2022 + 1][10 + 1] = {0};
    // dp[0][0] = 1;
    // for (int i = 1; i <= 2022; i++) {
    //     for (int j = 2022; j >= i; j--) {
    //         for (int k = 1; k <= 10; k++) {
    //             dp[j][k] += dp[j - i][k - 1];
    //         }
    //     }
    // }
    // return dp[2022][10];

    return ans;
}

int Solution::maxProfit(vector<int> &prices) {
    // 采用动态规划的方式dp[i][j],i表示第i天，j表示持有股票的状态
    // j共有两种情形
    int n = prices.size();
    vector<vector<int>> dp(n + 1, vector<int>(2));
    // 先进行初始化
    dp[0][0] = 0;
    dp[0][1] = -prices[0];

    for (int i = 1; i < n; i++) {
        dp[i][0] = std::max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
        // 区别于可以进行多次交易的递推公式
        dp[i][1] = std::max(dp[i - 1][1], -prices[i]);
    }
    return dp[n - 1][0];
}

int Solution::maxProfit_2(vector<int> &prices) {
    int n = prices.size();
    vector<vector<int>> dp(n + 1, vector<int>(2));
    dp[0][0] = 0;
    dp[0][1] = -prices[0];
    for (int i = 1; i < n; i++) {
        dp[i][0] = std::max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
        dp[i][1] = std::max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
    }
    return dp[n - 1][0];
}


int Solution::maxProfit_3(vector<int> &prices) {
    int n = prices.size();
    int K = 2;
    vector<vector<vector<int>>> dp(n + 1, vector<vector<int>>(2 + 1, vector<int>(2)));
    // 初始化dp Table,不管交易次数，第一天持有股票利润为-prices[0],不持有股票为0
    for (int k = 0; k < K + 1; k++) {
        dp[0][k][0] = 0;
        dp[0][k][1] = -prices[0];
    }
    for (int i = 1; i < n; i++) {
        for (int k = 1; k < K + 1; k++) {
            // 买入卖出算一次交易，这里只针对买入计算交易次数
            dp[i][k][0] = std::max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i]);
            dp[i][k][1] = std::max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i]);
        }
    }
    return dp[n - 1][K][0];
}


void backtracking(int n, int k, int startIndex, vector<vector<int>> &result, vector<int> &path) {
    if (path.size() == k) {
        result.push_back(path);
        return;
    }
    for (int i = startIndex; i <= n; i++) {
        path.push_back(i);// 处理节点
        backtracking(n, k, i + 1, result, path);// 递归
        path.pop_back();// 回溯，撤销处理的节点
    }
}
vector<vector<int>> Solution::combine(int n, int k) {
    vector<vector<int>> result;
    vector<int> path;
    backtracking(n, k, 1, result, path);
    return result;
}

void backtracking_combination_sum3(int n, int k, int start, vector<vector<int>> &result, vector<int> &stack) {
    if (stack.size() == k && std::accumulate(stack.begin(), stack.end(), 0) == n) {
        result.push_back(stack);
        return;
    }
    for (int i = start; i <= 9; i++) {
        stack.push_back(i);
        backtracking_combination_sum3(n, k, i + 1, result, stack);
        stack.pop_back();
    }
}

vector<vector<int>> Solution::combinationSum3(int k, int n) {
    vector<vector<int>> result;
    vector<int> stack;
    backtracking_combination_sum3(n, k, 1, result, stack);
    return result;
}

void backtracking_letter_combination(int start,
                                     string &digits,
                                     vector<string> &result,
                                     string &stack,
                                     string letters_map[10]) {
    if (stack.size() == digits.size()) {
        result.push_back(stack);
        return;
    }
    int digit      = digits[start] - '0';
    string letters = letters_map[digit];
    for (int j = 0; j < letters.size(); j++) {
        stack.push_back(letters[j]);
        backtracking_letter_combination(start + 1, digits, result, stack, letters_map);
        stack.pop_back();
    }
}

vector<string> Solution::letterCombinations(string digits) {
    vector<string> result;

    if (digits.size() == 0) return result;
    string letters_map[10] = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
    int n                  = digits.size();
    string stack;
    backtracking_letter_combination(0, digits, result, stack, letters_map);
    return result;
}

void backtracking_combination_sum(vector<int> &candidate,
                                  int target,
                                  int begin,
                                  vector<vector<int>> &result,
                                  vector<int> &stack) {
    if (target == 0) {
        result.push_back(stack);
        return;
    }
    if (target < 0) return;
    // 会产生排列结果，例如[[2,2,2,2],[2,3,3],[3,2,3],[3,3,2],[3,5],[5,3]]
    // for (int i : candidate) {
    //     stack.push_back(i);
    //     target -= i;
    //     backtracking_combination_sum(candidate, target, result, stack);
    //     // 回溯
    //     target += i;
    //     stack.pop_back();
    // }
    // 为避免出现排列结果，需要设置begin，不能再从头开始
    for (int i = begin; i < candidate.size(); i++) {
        stack.push_back(candidate[i]);
        target -= candidate[i];
        backtracking_combination_sum(candidate, target, i, result, stack);
        // 回溯
        target += candidate[i];
        stack.pop_back();
    }
}


vector<vector<int>> Solution::combinationSum(vector<int> &candidates, int target) {
    vector<vector<int>> result;
    vector<int> stack;
    backtracking_combination_sum(candidates, target, 0, result, stack);
    return result;
}

void backtracking_combination_sum_2(vector<pair<int, int>> &freq,
                                    int pos,
                                    int target,
                                    vector<vector<int>> &result,
                                    vector<int> &stack) {
    if (target == 0) {
        result.push_back(stack);
        return;
    }
    // 考虑边界条件需要考虑清楚，不然容易死循环
    if (target < 0 || pos == freq.size()) return;

    // 不选择当前数字
    backtracking_combination_sum_2(freq, pos + 1, target, result, stack);


    // 选择当前数字，先求可以最多选多少次

    int most = std::min(target / freq[pos].first, freq[pos].second);
    for (int i = 1; i <= most; i++) {
        stack.push_back(freq[pos].first);
        target -= freq[pos].first;
        backtracking_combination_sum_2(freq, pos + 1, target, result, stack);
    }
    for (int j = 1; j <= most; j++) {
        // target 回溯
        target += freq[pos].first;
        // stack 回溯
        stack.pop_back();
    }
}

vector<vector<int>> Solution::combinationSum2(vector<int> &candidates, int target) {
    // 方案一：考虑如何在递归的过程中进行去重
    // 方案二：先将candidates进行分组，然后再回溯
    vector<vector<int>> result;
    vector<int> stack;
    std::sort(candidates.begin(), candidates.end());
    vector<pair<int, int>> freq;
    for (int num : candidates) {
        if (freq.empty() || num != freq.back().first) {
            freq.emplace_back(num, 1);
        } else {
            ++freq.back().second;
        }
    }
    backtracking_combination_sum_2(freq, 0, target, result, stack);
    return result;
}

bool is_palindrome(const string &s, int start, int end) {
    for (int i = start, j = end; i < j; i++, j--) {
        if (s[i] != s[j]) { return false; }
    }
    return true;
}
void backtracking_partition(const string &s, int pos, vector<vector<string>> &result, vector<string> &stack) {
    // 递归的结束条件
    if (pos >= s.size()) {
        result.push_back(stack);
        return;
    }

    for (int i = pos; i < s.size(); i++) {
        string t = s.substr(pos, i - pos + 1);
        if (is_palindrome(t, 0, t.size() - 1)) {
            // 如果是回文串，我们再开始递归
            stack.push_back(t);
            backtracking_partition(s, pos + t.size(), result, stack);
            stack.pop_back();
        }
    }
}

vector<vector<string>> Solution::str_partition(string s) {
    vector<vector<string>> result;
    vector<string> stack;

    backtracking_partition(s, 0, result, stack);
    return result;
}

bool is_valid_ip_address(const string &s) {
    if (s.size() == 0) return false;
    if (s.size() > 1 && s[0] == '0') { return false; }
    int num = 0;
    if (s.size() > 1) {
        for (int i = 0; i < s.size(); i++) {
            if (s[i] > '9' || s[i] < '0') { return false; }
            num = num * 10 + (s[i] - '0');
            if (num > 255) return false;
        }
        // 值不能大于255
    }
    return true;
}

void backtracking_restore_ip_addresses(const string &s, int pos, vector<string> &result, vector<string> &stack) {
    // 递归的终止条件
    if (pos >= s.size()) {
        if (stack.size() == 4) {
            string o = "";
            for (int i = 0; i < stack.size() - 1; i++) {
                o += stack[i];
                o += ".";
            }
            o += stack[stack.size()];

            result.emplace_back(o);
        }
        return;
    }

    for (int i = pos; i < s.size(); i++) {
        string t = s.substr(pos, i - pos + 1);
        if (is_valid_ip_address(t)) {
            stack.emplace_back(t);
            backtracking_restore_ip_addresses(s, pos + t.size(), result, stack);
            stack.pop_back();
        }
    }
}


vector<string> Solution::restoreIpAddresses(string s) {
    vector<string> result;
    vector<string> stack;
    backtracking_restore_ip_addresses(s, 0, result, stack);
    return result;
}


void backtracking_subsets(vector<int> &nums, int pos, vector<vector<int>> &result, vector<int> &stack) {
    result.push_back(stack);
    for (int i = pos; i < nums.size(); i++) {
        stack.push_back(nums[i]);
        backtracking_subsets(nums, i + 1, result, stack);
        stack.pop_back();
    }
}

vector<vector<int>> Solution::subsets(vector<int> &nums) {
    vector<vector<int>> result;
    vector<int> stack;
    int pos = 0;
    backtracking_subsets(nums, pos, result, stack);

    return result;
}


void backtracking_subsets_dup(vector<int> &nums,
                              int pos,
                              vector<vector<int>> &result,
                              vector<bool> &used,
                              vector<int> stack) {
    result.push_back(stack);
    for (int i = pos; i < nums.size(); i++) {
        // 保证再递归在同一深度的位置，不能采样相同的数据，如果数据相同，势必会出现重复的结果（可以这么理解，如果i-1==i,那么第i-1次
        // 的递归结果必然包含在第i的递归结果中）
        // used[i - 1] == true，说明同一树支candidates[i - 1]使用过
        // used[i - 1] == false，说明同一树层candidates[i - 1]使用过,
        if (i > 0 && nums[i] == nums[i - 1] && used[i - 1] == false) { continue; }
        stack.push_back(nums[i]);
        used[i] = true;
        backtracking_subsets_dup(nums, i + 1, result, used, stack);
        used[i] = false;
        stack.pop_back();
    }
}


vector<vector<int>> Solution::subsetsWithDup(vector<int> &nums) {
    // 采用used 数组来进行去重，不同于先进行分组，再来递归
    vector<vector<int>> result;
    vector<int> stack;
    // 采用used数组,初始化为false
    vector<bool> used(nums.size(), false);

    // 排序
    std::sort(nums.begin(), nums.end());
    backtracking_subsets_dup(nums, 0, result, used, stack);


    return result;
}

void backtracking_find_subsequences(vector<int> &nums, int pos, vector<vector<int>> &result, vector<int> &stack) {
    // 至少需要两个元素
    if (stack.size() > 1) { result.push_back(stack); }

    std::unordered_set<int> set;
    for (int i = pos; i < nums.size(); i++) {
        // 1.递增
        // 2.去重
        if (stack.empty()) {
            if (set.find(nums[i]) == set.end()) {
                stack.push_back(nums[i]);
                set.insert(nums[i]);
                backtracking_find_subsequences(nums, i + 1, result, stack);
                stack.pop_back();
            }
        } else {
            if (nums[i] >= stack.back()) {
                if (set.find(nums[i]) == set.end()) {
                    // 不在集合中
                    set.insert(nums[i]);
                    stack.push_back(nums[i]);
                    backtracking_find_subsequences(nums, i + 1, result, stack);
                    stack.pop_back();
                }
            }
        }
        // if (set.find(nums[i]) != set.end()) {
        //     if (nums[i] >= stack.back()) stack.push_back(nums[i]);
        //     set.insert(nums[i]);
        //     backtracking_find_subsequences(nums, i + 1, result, stack);
        //     stack.pop_back();
        // }
    }
}

vector<vector<int>> Solution::findSubsequences(vector<int> &nums) {
    vector<vector<int>> result;
    vector<int> stack;
    backtracking_find_subsequences(nums, 0, result, stack);
    return result;
}

void backtracking_permute(vector<int> nums, vector<bool> &used, vector<vector<int>> &result, vector<int> &stack) {
    if (stack.size() == nums.size()) result.push_back(stack);

    for (int i = 0; i < nums.size(); i++) {
        if (used[i] == false) {
            stack.push_back(nums[i]);
            used[i] = true;
            backtracking_permute(nums, used, result, stack);
            stack.pop_back();
            used[i] = false;
        }
    }
}

vector<vector<int>> Solution::permute(vector<int> &nums) {
    vector<vector<int>> result;
    vector<int> stack;
    vector<bool> used(nums.size(), false);
    backtracking_permute(nums, used, result, stack);
    return result;
}


void backtracking_permute_unique(vector<int> &nums,
                                 vector<bool> &used,
                                 vector<vector<int>> &result,
                                 vector<int> &stack) {
    if (stack.size() == nums.size()) {
        result.push_back(stack);
        return;
    }
    for (int i = 0; i < nums.size(); i++) {
        if (stack.empty()) {
            if (used[i] == false) {
                stack.push_back(nums[i]);
                used[i] = true;
                backtracking_permute_unique(nums, used, result, stack);
                stack.pop_back();
                used[i] = false;
            }
        } else {
            if (used[i] == false) {
                if (i == 0 || nums[i] != nums[i - 1] || used[i - 1] == false) {
                    // 注意i==0的位置，需要优先判断，不然会出现数组越界的问题
                    stack.push_back(nums[i]);
                    used[i] = true;
                    backtracking_permute_unique(nums, used, result, stack);
                    stack.pop_back();
                    used[i] = false;
                }
            }
        }
    }
}

vector<vector<int>> Solution::permuteUnique(vector<int> &nums) {
    vector<vector<int>> result;
    vector<int> stack;
    vector<bool> used(nums.size(), false);
    std::sort(nums.begin(), nums.end());
    backtracking_permute_unique(nums, used, result, stack);
    return result;
}

vector<string> hierholzer_find_itinerary(vector<vector<string>> &tickets, string &vertex) {
    vector<string> result;
    // 先构建图(图的存储方案),欧拉半图
    std::unordered_map<string, std::priority_queue<string, vector<string>, std::greater<string>>> g;
    for (auto t : tickets) { g[t[0]].push(t[1]); }

    std::stack<string> cpath;
    std::stack<string> epath;
    cpath.push(vertex);
    while (!cpath.empty()) {
        string u = cpath.top();
        if (g[u].empty()) {
            epath.push(u);
            cpath.pop();
        } else {
            cpath.push(g[u].top());
            g[u].pop();
        }
    }
    while (!epath.empty()) {
        result.push_back(epath.top());
        epath.pop();
    }
    return result;
}

vector<string> Solution::findItinerary(vector<vector<string>> &tickets) {
    // 注意区分dfs+回溯 与 Hierholzer算法的区别
    // Hierholzer算法：具体参考 https://slaystudy.com/hierholzers-algorithm/
    string start = "JFK";
    return hierholzer_find_itinerary(tickets, start);
}


vector<vector<string>> Solution::solveNQueens(int n) {
    vector<vector<string>> result;
    // 用来记录第i行皇后在哪一列
    vector<int> stack(n, 0);

    std::function<bool(int, int)> is_valid = [&](int row, int column) {
        // 第零行随便放
        if (row == 0) return true;
        for (int R = 0; R < row; R++) {
            int C = stack[R];
            // 不能处在同一列
            if (column == C) return false;
            // 不能再同一斜列
            if ((row + column) == (R + C) || (R - C) == (row - column)) { return false; }
        }
        return true;
    };

    std::function<void(int)> dfs_solve_n_queens = [&](int row) {
        // row 表示行，column表示列
        if (row == n) {
            // 所有皇后已经放好,返回条件
            vector<string> board(n);
            for (int i = 0; i < stack.size(); ++i) {
                // 注意单引号和双引号的区别
                board[i] = string(stack[i], '.') + 'Q' + string(n - 1 - stack[i], '.');
            }
            result.emplace_back(board);
        }
        for (int c = 0; c < n; ++c) {
            // 判断当前位置第row行，第c列能否放置皇后
            if (is_valid(row, c)) {
                stack[row] = c;
                // 如果找到合适的位置，row才会加1
                dfs_solve_n_queens(row + 1);
            }
            // 若当前位置未找到有效位置，row无法递增，也就是无法递归，result为初始化结果
        }
    };
    dfs_solve_n_queens(0);

    return result;
}

void Solution::solveSudoku(vector<vector<char>> &board) {
    // 注意容易超时

    // 用于判断当前位置填入的数是否有效
    std::function<bool(int row, int column, char c)> is_valid = [&](int row, int column, char c) {
        // 先判断是数字0-9
        if ((c - '0') <= 9 && (c - '0') > 0) {
            // 行
            for (int n = 0; n < board[row].size(); ++n) {
                if (c == board[row][n]) { return false; }
            }
            // 列
            for (int n = 0; n < board.size(); n++) {
                if (c == board[n][column]) { return false; }
            }
            // 3*3的正方形中只能出现一次
            // 先求解行列的范围
            int R = (row / 3) * 3;
            int C = (column / 3) * 3;
            for (int i = R; i < R + 3; i++) {
                for (int j = C; j < C + 3; j++) {
                    if (c == board[i][j]) return false;
                }
            }
            return true;

        } else {
            return false;
        }
    };

    std::function<bool(int row, int column)> backtracking_solve_sudoku = [&](int row, int column) {
        char value[] = "123456789";
        if (column == board[0].size()) {
            // 需要注意便利顺序
            row++;
            column = 0;
            if (row == board.size()) { return true; }
        }
        if (board[row][column] != '.') {
            // 直接跳过当前位置,前进一位
            return backtracking_solve_sudoku(row, column + 1);
        }
        for (int i = 0; i < 9; i++) {
            if (is_valid(row, column, value[i])) {
                board[row][column] = value[i];
                if (backtracking_solve_sudoku(row, column + 1)) return true;
                // 错误需要回溯
                board[row][column] = '.';
            }
        }
        return false;
    };
    backtracking_solve_sudoku(0, 0);
}

vector<int> Solution::preorderTraversal(TreeNode *root) {
    // 递归方式实现
    vector<int> result;
    if (root == nullptr) return result;
    std::function<void(TreeNode *)> dfs = [&](TreeNode *cur) {
        if (cur == NULL) return;
        result.emplace_back(cur->val);
        dfs(cur->left);
        dfs(cur->right);
    };
    dfs(root);

    return result;
}

vector<int> Solution::inorderTraversal(TreeNode *root) {
    // 递归式解法
    // vector<int> result;
    // if (root == nullptr) return result;

    // std::function<void(TreeNode *)> dfs = [&](TreeNode *cur) {
    //     // 确定递归的终止条件
    //     if (cur == nullptr) return;
    //     dfs(cur->left);
    //     result.emplace_back(cur->val);
    //     dfs(cur->right);
    // };

    // dfs(root);
    // return result;

    // 迭代式解法
    vector<int> result;
    if (root == nullptr) return result;
    std::stack<TreeNode *> st;
    // 当前的节点指针
    TreeNode *cur = root;
    // 主要原因是数据处理 和数据访问不一致，我们总是从根节点进行访问，但是中序确实先处理左节点
    // 注意栈的尚未初始化，外层循环需要添加限制项
    while (!st.empty() || cur != nullptr) {
        while (cur != nullptr) {
            // 记录访问路径
            st.push(cur);
            cur = cur->left;
        }
        // 左子树为空的时候，并非达到达叶节点的时候,无法继续向左递归的时候
        cur = st.top();
        // 处理当前节点
        result.emplace_back(cur->val);
        st.pop();
        cur = cur->right;
    }
    return result;
}

vector<int> Solution::postorderTraversal(TreeNode *root) {
    // 利用栈,采用迭代的方式
    vector<int> result;
    if (root == nullptr) return result;
    std::stack<TreeNode *> st;
    st.push(root);
    while (!st.empty()) {
        TreeNode *top = st.top();
        result.emplace_back(top->val);
        st.pop();
        if (top->left) st.push(top->left);
        if (top->right) st.push(top->right);
    }

    std::reverse(result.begin(), result.end());
    return result;
}


vector<vector<int>> Solution::levelOrder(TreeNode *root) {
    vector<vector<int>> result;
    if (root == nullptr) return result;
    std::queue<TreeNode *> que;
    que.push(root);
    while (!que.empty()) {
        vector<int> path;
        int num = que.size();
        // 需要提前计算好que的大小，因为que的size会变化
        for (int i = 0; i < num; i++) {
            TreeNode *node = que.front();
            path.emplace_back(node->val);
            que.pop();
            // 非空的时候，添加到队列
            if (node->left) que.push(node->left);
            if (node->right) que.push(node->right);
        }
        result.emplace_back(path);
    }
    return result;
}

vector<vector<int>> Solution::levelOrderBottom(TreeNode *root) {
    vector<vector<int>> result;
    if (root == nullptr) return result;
    std::queue<TreeNode *> que;
    que.push(root);
    while (!que.empty()) {
        vector<int> path;
        int num = que.size();
        // 需要提前计算好que的大小，因为que的size会变化
        for (int i = 0; i < num; i++) {
            TreeNode *node = que.front();
            path.emplace_back(node->val);
            que.pop();
            // 非空的时候，添加到队列
            if (node->left) que.push(node->left);
            if (node->right) que.push(node->right);
        }
        result.emplace_back(path);
    }
    std::reverse(result.begin(), result.end());
    return result;
}


vector<int> Solution::rightSideView(TreeNode *root) {
    vector<int> result;
    if (root == nullptr) return result;
    std::queue<TreeNode *> que;
    que.push(root);
    while (!que.empty()) {
        vector<int> path;
        int num = que.size();
        // 需要提前计算好que的大小，因为que的size会变化
        for (int i = 0; i < num; i++) {
            TreeNode *node = que.front();
            path.emplace_back(node->val);
            que.pop();
            // 非空的时候，添加到队列
            if (node->left) que.push(node->left);
            if (node->right) que.push(node->right);
        }
        result.emplace_back(path.back());
    }

    return result;
}

vector<double> Solution::averageOfLevels(TreeNode *root) {
    vector<double> result;
    if (root == nullptr) return result;
    std::queue<TreeNode *> que;
    que.push(root);
    while (!que.empty()) {
        double sum = 0;
        int nums   = que.size();
        for (int i = 0; i < nums; i++) {
            TreeNode *node = que.front();
            sum += node->val;
            que.pop();
            if (node->left) que.push(node->left);
            if (node->right) que.push(node->right);
        }
        double average = sum / nums;
        result.emplace_back(average);
    }
    return result;
}

TreeNode *Solution::invertTree(TreeNode *root) {

    // // 需要考虑清楚终止条件
    // if (root == nullptr) return root;
    // std::function<void(TreeNode *)> invert = [&](TreeNode *node) {
    //     TreeNode *temp = node->left;
    //     node->left     = node->right;
    //     node->right    = temp;
    //     if (node->left) invert(node->left);
    //     if (node->right) invert(node->right);
    // };
    // invert(root);

    // return root;

    // 采用迭代的方式进行尝试
    if (root == nullptr) return root;
    std::stack<TreeNode *> st;
    st.push(root);
    while (!st.empty()) {
        TreeNode *node = st.top();
        TreeNode *temp = node->left;
        node->left     = node->right;
        node->right    = temp;
        st.pop();
        if (node->right) st.push(node->right);
        if (node->left) st.push(node->left);
    }
    return root;
}
bool Solution::isSymmetric(TreeNode *root) {
    if (root == nullptr) return true;
    // 需要递归判断左右子树是个否相等,核心是后序遍历
    std::function<bool(TreeNode *, TreeNode *)> dfs_compare = [&](TreeNode *left, TreeNode *right) {
        if (left == nullptr && right != nullptr)
            return false;
        else if (left != nullptr && right == nullptr)
            return false;
        else if (left == nullptr && right == nullptr)
            return true;
        else if (left->val != right->val)
            return false;

        bool l = dfs_compare(left->left, right->right);
        bool r = dfs_compare(left->right, right->left);

        return l && r;
    };

    return dfs_compare(root->left, root->right);
    // 采用非递归的方式,类似层序遍历,但是需要注意,进入队列的顺序
}

int Solution::maxDepth(TreeNode *root) {
    // 采用迭代的方式计算,层序遍历
    if (root == nullptr) return 0;
    int depth = 0;
    std::queue<TreeNode *> que;
    que.push(root);
    while (!que.empty()) {
        int size = que.size();
        for (int i = 0; i < size; ++i) {
            TreeNode *node = que.front();
            que.pop();
            if (node->left) que.push(node->left);
            if (node->right) que.push(node->right);
        }
        depth++;
    }
    return depth;
}

int Solution::maxDepth_N(Node *root) {
    if (root == nullptr) return 0;
    int depth = 0;
    std::queue<Node *> que;
    que.push(root);
    while (!que.empty()) {
        int size = que.size();
        for (int i = 0; i < size; ++i) {
            Node *node = que.front();
            que.pop();
            for (int j = 0; j < node->children.size(); j++) {
                if (node->children[j]) que.push(node->children[j]);
            }
        }
        depth++;
    }
    return depth;
}

int Solution::minDepth(TreeNode *root) {
    if (root == nullptr) return 0;
    std::function<int(TreeNode *)> dfs = [&](TreeNode *node) {
        // 如果当前节点为空,最小高度为0
        if (node == nullptr) return 0;
        if (node->left == nullptr && node->right != nullptr) {
            int right = dfs(node->right);
            return right + 1;
        } else if (node->right == nullptr && node->left != nullptr) {
            int left = dfs(node->left);
            return left + 1;
        } else if (node->left != nullptr && node->right != nullptr) {
            int left  = dfs(node->left);
            int right = dfs(node->right);
            // 后序遍历的体现
            return std::min(left, right) + 1;
        } else {
            return 1;
        }
    };
    return dfs(root);
}

bool Solution::isBalanced(TreeNode *root) {
    // 重点是如何构建递归函数的返回值,便于后序遍历计算机高度差
    std::function<int(TreeNode *)> get_depth = [&](TreeNode *node) {
        if (node == nullptr) return 0;
        // 递归终止条件

        int left_depth = get_depth(node->left);
        if (left_depth == -1) return -1;
        int right_depth = get_depth(node->right);
        if (right_depth == -1) return -1;
        if (std::abs(left_depth - right_depth) > 1) return -1;
        return 1 + std::max(left_depth, right_depth);
    };

    return get_depth(root) != -1;
}


bool Solution::isValidBST(TreeNode *root) {
    // 利用中序遍历,判断递增
    // prev初始化需要设置
    // TODO 梳理下节点非空的判断场景
    TreeNode *prev                           = nullptr;
    std::function<bool(TreeNode * node)> dfs = [&](TreeNode *node) -> bool {
        if (node == nullptr) return true;
        bool left = dfs(node->left);
        if (prev != nullptr && node->val <= prev->val) return false;
        prev       = node;
        bool right = dfs(node->right);
        return left && right;
    };
    return dfs(root);
}

vector<string> binaryTreePaths(TreeNode *root) {
    //  这里注意边界条件
    vector<string> result;
    vector<int> path;
    if (root == nullptr) return result;
    std::function<void(TreeNode *)> dfs = [&](TreeNode *node) {
        path.push_back(node->val);

        if (node->left == nullptr && node->right == nullptr) {
            // 叶节点判定,注意
            string o = "";
            for (int i = 0; i < path.size() - 1; i++) {
                o += std::to_string(path[i]);
                o += "->";
            }
            o += std::to_string(path[path.size() - 1]);
            result.push_back(o);
            return;
        }
        // 没有在前面进行非空判断,需要在输入的时候进行非空判断
        if (node->left) {
            dfs(node->left);
            path.pop_back();
        }
        if (node->right) {
            dfs(node->right);
            path.pop_back();
        }
    };
    dfs(root);
    return result;
    // 采用迭代的方式,第一种把到所有节点的路径都记录下来，可以用一个栈模拟递归，一个栈来记录路径
}
vector<int> Solution::findDisappearedNumbers(vector<int> &nums) {
    vector<int> result;
    for (int i = 0; i < nums.size(); i++) {
        int pos = std::abs(nums[i]) - 1;
        if (nums[pos] > 0) { nums[pos] = -nums[pos]; }
    }
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] > 0) { result.emplace_back(nums[i] + 1); }
    }
    return result;
}

string Solution::removeDuplicates(string s) {
    std::stack<char> st;
    vector<char> result;
    for (char c : s) {
        if (!st.empty() && c == st.top()) {
            st.pop();
        } else {
            st.push(c);
        }
    }
    while (!st.empty()) {
        result.push_back(st.top());
        st.pop();
    }

    std::reverse(result.begin(), result.end());
    string output;
    for (int i = 0; i < result.size(); i++) { output += result[i]; }
    return output;
}
vector<string> Solution::topKFrequent(vector<string> &words, int k) {
    vector<string> result(k);
    std::unordered_map<string, int> freq;
    for (string &s : words) {
        // 为什么用引用
        freq[s]++;
    }

    auto compare = [](std::pair<string, int> &a, std::pair<string, int> &b) {
        return a.second == b.second ? a.first < b.first : a.second > b.second;
    };
    std::priority_queue<std::pair<string, int>, vector<std::pair<string, int>>, decltype(compare)> que(compare);
    for (auto &p : freq) {
        que.emplace(std::pair<string, int>(p.first, p.second));
        if (que.size() > k) { que.pop(); }
    }
    for (int i = 0; i < k - 1; i++) {
        result[k-i-i]=que.top().first;
        que.pop();
    }
    return result;
}

}// namespace leetcode
