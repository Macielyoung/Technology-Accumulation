#-*- coding: UTF-8 -*-

class DynamicProgramming(object):
    # 剪绳子问题，把长为n的生子剪成m段(每部分都为整数)，求这些数值的最大乘积
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

    # 现有1，3，5三种面额的硬币若干枚，如何使用最少的硬币组成金额n
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

if __name__ == "__main__":
    DP = DynamicProgramming()

    # 剪绳子问题
    n_slot = 8
    slot_res = DP.splitSlot(n_slot)
    print(slot_res)

    # 分硬币问题
    coin_num = 11
    coin_res = DP.collectCoins(coin_num)
    print(coin_res)






