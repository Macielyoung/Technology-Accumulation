#-*- coding: UTF-8 -*-

class DynamicProgramming(object):
    # 1.剪绳子问题，把长为n的生子剪成m段(每部分都为整数)，求这些数值的最大乘积
    def splitSlot(self, n):
        # 讨论特殊情况（即最基础情况）
        if n < 2:
            return 0
        if n == 2:
            return 1
        if n == 3:
            return 2
        dp = [0 for i in range(n+1)]
        dp[1], dp[2], dp[3] = 1, 2, 3
        for i in range(4, n+1):
            maxlen = 0
            for j in range(1, i/2+1):
                length = dp[j] * dp[i-j]
                if length > maxlen:
                    maxlen = length
                dp[i] = maxlen
        return dp[-1]

    # 2.现有1，3，5三种面额的硬币若干枚，如何使用最少的硬币组成金额n
    def collectCoins(self, n):
        if n == 1 or n == 3 or n == 5:
            return 1
        if n == 2 or n == 4:
            return 2
        dp = [0 for i in range(n+1)]
        dp[1], dp[3], dp[5] = 1, 1, 1
        dp[2], dp[4] = 2, 2
        for i in range(6, n+1):
            minnum = i
            for j in [1,3,5]:
                if dp[i-j]+1<minnum:
                    minnum = dp[i-j]+1
            dp[i] = minnum
        return dp[-1]

    # 3.现有数量不限的1，5，10，25四种面额的硬币，表示金额为n有多少种方法
    # 可以构建一个二维表ways[i][j],表示拆分金额j分使用前i种面额的方法数
    # 则增加一种面额有两种情况：1.ways[i][j]=ways[i-1][j] 2.ways[i][j]=ways[i-1][j]+ways[i][j-coins[i]]
    def changeCoins(self, coins, n):
        m = len(coins)
        if m == 0 or n < 0:
            return 0
        ways = [[0 for j in range(n+1)] for i in range(m)]
        for i in range(m):
            ways[i][0] = 1
        for j in range(1, n+1):
            ways[0][j] = 1
        for i in range(1, m):
            for j in range(1, n+1):
                if coins[i] > j:
                    ways[i][j] = ways[i-1][j]
                else:
                    ways[i][j] = ways[i-1][j] + ways[i][j-coins[i]]
        return ways[-1][-1]

    # 4.爬楼梯问题，一只青蛙每次可以爬一阶或两阶楼梯，问爬n阶楼梯有多少种方法
    def climbStairs(self, n):
        if n <= 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 2
        stairs = [0 for i in range(n+1)]
        stairs[1], stairs[2] = 1, 2
        for i in range(3, n+1):
            stairs[i] = stairs[i-1] + stairs[i-2]
        return stairs[-1]

    # 5.装箱问题/背包问题
    def backPacking(self, packages, V):
        if not packages or V <= 0:
            return 0
        n = len(packages)
        capcity = [[0 for i in range(V+1)] for j in range(n+1)]
        # c[n][V]表示前n件物品转入箱子中最大占据的空间
        for i in range(1, n+1):
            for j in range(1, V+1):
                # 如果剩余的空间大于等于物品容量，那物品可放可不放（比较放入和不放入两种情况下占据的空间大小）
                if j >= packages[i-1]:
                    capcity[i][j] = max(capcity[i-1][j], capcity[i-1][j-packages[i-1]]+packages[i-1])
                # 如果剩余的空间小于物品容量，那物品则不能放入
                else:
                    capcity[i][j] = capcity[i-1][j]
        return capcity[-1][-1]

    # 6.最长递增子序列问题
    def longSubSequence(self, nums):
        # 时间复杂度为O(n^2)
        # 每一个数都和它前面的数做比较
        if not nums:
            return 0
        n = len(nums)
        length = [1 for i in range(n)]
        for i in range(n-1):
            for j in range(i+1):
                if nums[i+1] > nums[j]:
                    length[i+1] = max(length[i+1], length[j]+1)
        return max(length)

    # 时间复杂度O(nlogn)，按dp[t]=k来分类，只要保留dp[t]中最小的nums[t].
    def LIS(self, nums):
        def binarySearch(key, g, low, high):
            while(low < high):
                mid = (low + high) >> 1
                if key >= g[mid]:
                    low = mid + 1
                else:
                    high = mid
            return low
        n, j = len(nums), 0
        g = [0 for i in range(n)]
        g[1], length = nums[0], 1
        for i in range(1, n):
            if g[length] < nums[i]:
                length += 1
                j = length
            else:
                j = binarySearch(nums[i], g, 1, length+1) # 二分查找，找到第一个比nums[i]小的g[k]
            g[j] = nums[i]
        return length

    # 7.最长公共子序列问题
    def longCommonSequence(self, arr1, arr2):
        # 时间复杂度：O（m*n）
        # 使用dp[i][j]表示字符串A第i个位置和字符串第j个位置结尾的最长公共子序列
        # if i==0 or j==0, dp[i][j]=0
        # if A[i]==B[j], dp[i][j]=dp[i-1][j-1]+1
        # if A[i]!=B[j], dp[i][j]=max(dp[i-1][j], dp[i][j-1])
        l1, l2 = len(arr1), len(arr2)
        dp = [[0]*(l2+1)]*(l1+1)
        for i in range(1, l1+1):
            for j in range(1, l2+1):
                if arr1[i-1] == arr2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

    # 8.最长公共子串问题
    def LCstring(self, string1, string2):
        # 时间复杂度：O（m*n）
        # res[i][j]表示以A[i]和B[j]为结尾元素的最长公共子串长度
        # i==0 or j==0, res[i][j] = 0
        # if A[i]==B[j], res[i][j] = res[i-1][j-1] + 1
        # if A[i]!=B[j], res[i][j] = 0
        len1 = len(string1)
        len2 = len(string2)
        res = [[0 for i in range(len1+1)] for j in range(len2+1)]
        result = 0
        end = 0
        for i in range(1,len2+1):
            for j in range(1,len1+1):
                if string2[i-1] == string1[j-1]:
                    res[i][j] = res[i-1][j-1]+1
                    if res[i][j] > result:
                        result = res[i][j]
                        end = i
        common_substr = string2[end-result: end] 
        return result, common_substr

    # 9.最大连续子序列和问题
    # 使用maxSum[i]来表示以第i个数字结尾的子数组的最大和，需要求出max(maxSum[i])
    def maxSequenceSum(self, nums):
        if not nums:
            return 0
        n = len(nums)
        maxSum = [0 for i in range(n)]
        maxSum[0], start = nums[0], 0
        for i in range(1, n):
            if maxSum[i-1] <= 0:
                maxSum[i] = nums[i]
            else:
                maxSum[i] = maxSum[i-1]+nums[i]
        return max(maxSum)

    # 10.股票交易最大化（一次交易）
    def maxProfit_onedeal(self, prices):
        n = len(prices)
        if n < 2:
            return 0
        minPrice = prices[0]
        maxProfit = prices[1] - prices[0]
        for i in range(2, n):
            if prices[i-1] < minPrice:
                minPrice = prices[i-1]
            profit = prices[i] - minPrice
            if profit > maxProfit:
                maxProfit = profit
        return maxProfit if maxProfit > 0 else 0


if __name__ == "__main__":
    DP = DynamicProgramming()

    # # 剪绳子问题
    # n_slot = 8
    # slot_res = DP.splitSlot(n_slot)
    # print(slot_res)
    #
    # # 分硬币问题
    # coin_num = 11
    # coin_res = DP.collectCoins(coin_num)
    # print(coin_res)
    #
    # # 换硬币问题
    # coins = [1, 5, 10, 25]
    # n_coin = 10
    # coin_ways = DP.changeCoins(coins, n_coin)
    # print(coin_ways)
    #
    # # 爬楼梯问题
    # stair_num = 3
    # stair_ways = DP.climbStairs(stair_num)
    # print(stair_ways)
    #
    # # 背包问题
    # packages = [6, 8, 3, 12, 7, 9]
    # V = 24
    # max_capcity = DP.backPacking(packages, V)
    # print(max_capcity)
    #
    # # 最长上升子序列问题
    # nums6 = [1, 6, 2, 3, 7, 5]
    # longest = DP.longSubSequence(nums6)
    # longest2 = DP.LIS(nums6)
    # print(longest, longest2)

    # 最大连续子数组和
    # nums9 = [-2, 11, -4, 13, -5, -2]
    # maxSum = DP.maxSequenceSum(nums9)
    # print(maxSum)

    # 股票交易问题(一次交易)
    # prices_one = [9,11,8,5,7,12,16,14]
    # maxProfit_one = DP.maxProfit_onedeal(prices_one)
    # print(maxProfit_one)

    # 最长公共子串长度
    # str1 = "1AB2345CD"
    # str2 = "12345EF"
    # res, common_substr = DP.LCstring(str1, str2)
    # # res, common_substr = DP.longest_sublist_length(listA, listB)
    # print(res)
    # print(common_substr)

    arr1 = "helloworld"
    arr2 = "loop"
    res = DP.longCommonSequence(arr1, arr2)
    print(res)