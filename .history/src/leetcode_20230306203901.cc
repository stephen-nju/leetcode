
#include "leetcode.h"

#include <cstdlib>
namespace leetcode {

int Solution::minDistance(string word1, string word2) {
    int distance = 0;
    int m = word1.size();
    int n = word2.size();
    int **dp = (int **)malloc(sizeof(int *) * (m + 1));
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
    vector<vector<bool> > dp;
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

    int left_depth = maxDepth(root->left, diameter);
    int right_depth = maxDepth(root->right, diameter);
    *diameter = std::max(right_depth + left_depth, *diameter);
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
    temp = arr[x];
    arr[x] = arr[y];
    arr[y] = temp;
}

int partition(int arr[], int left, int right) {
    // 随机选择pivot
    int pivot_index = left + rand() % (right - left + 1);
    int pivot = arr[pivot_index];
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
    vector<vector<int> > dp(n + 1, vector<int>(2));
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
    vector<vector<int> > dp(n + 1, vector<int>(2));
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
    vector<vector<vector<int> > > dp(n + 1, vector<vector<int> >(2 + 1, vector<int>(2)));
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


/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */

void backtracking(int cur, int n, int k, int **ans, int *ans_size, int *temp, int temp_size) {
    if (temp_size == k) {
        int *tmp = (int *)malloc(sizeof(int) * (k + 1));
        for (int i = 0; i < k; i++) { tmp[i] = temp[i]; }
        printf("%d",*ans_size);
        ans[(*ans_size)++] = tmp;
        return;
    }
    for(int i=cur;i<n;i++){
    temp[temp_size++] = cur;
    // printf("%d",cur);
    backtracking(i+1, n, k, ans, ans_size, temp, temp_size);
    temp_size--;
    }
}

int **Solution::combine(int n, int k, int *returnSize, int **returnColumnSizes) {
    int **ans = (int **)malloc(sizeof(int *) * 10001);
    int *temp = (int *)malloc(sizeof(int) * (k + 1));
    int* ans_size = 0;
    int temp_size = 0;
    backtracking(1, n, k, ans, ans_size, temp, temp_size);
    *returnSize = ans_size;
    printf("%d",ans_size);
    *returnColumnSizes = (int *)malloc(sizeof(int) * ans_size);
    for (int i = 0; i < ans_size; i++) { (*returnColumnSizes)[i] = k; }
    return ans;
}


}// namespace leetcode
