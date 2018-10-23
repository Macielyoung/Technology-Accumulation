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

    # 现有数量不限的1，5，10，25四种面额的硬币，表示金额为n有多少种方法
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

    # 爬楼梯问题，一只青蛙每次可以爬一阶或两阶楼梯，问爬n阶楼梯有多少种方法
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

    # 装箱问题/背包问题
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

    # 换硬币问题
    coins = [1, 5, 10, 25]
    n_coin = 10
    coin_ways = DP.changeCoins(coins, n_coin)
    print(coin_ways)

    # 爬楼梯问题
    stair_num = 3
    stair_ways = DP.climbStairs(stair_num)
    print(stair_ways)

    # 背包问题
    packages = [6, 8, 3, 12, 7, 9]
    V = 24
    max_capcity = DP.backPacking(packages, V)
    print(max_capcity)