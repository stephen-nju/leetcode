
#include "leetcode.h"

#include <cstdlib>
#include <numeric>
#include <unordered_set>
#include <utility>
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

int maxDepth(TreeNode *root, int *diameter) {
    if (root == nullptr) return 0;

    int left_depth  = maxDepth(root->left, diameter);
    int right_depth = maxDepth(root->right, diameter);
    *diameter       = std::max(right_depth + left_depth, *diameter);
    return std::max(left_depth, right_depth) + 1;
}

int Solution::diameterOfBinaryTree(TreeNode *root) {
    int diameter = 0;
    if (root == nullptr) { return 0; }
    maxDepth(root, &diameter);
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
        // used[i - 1] == false，说明同一树层candidates[i - 1]使用过
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
            used[i] == true;
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

}// namespace leetcode
