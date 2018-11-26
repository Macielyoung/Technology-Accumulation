# coding:utf-8

# @Created : Macielyoung
# @Time : 2018/11/20
# @Function : Traverse a two-dimensional matrix(only including 0&1), and get rows which have the most 1.

# 题目要求：
# 输入是一个只含有 0 和 1 的二维矩阵，每一行都是排过序的，
# 也就是说每一行前一部分都是 0,剩下的全都是 1。
# 请找哪些行包含的 1 最多。
# 要求对于 MxN 的矩阵，时间复杂度是 O(M+N)，空间复杂度是 O(1)

# 示例:
# [0 0 0 0 0 0 0 1 1 1 1 1]
# [0 0 0 0 1 1 1 1 1 1 1 1]
# [0 0 0 0 0 0 1 1 1 1 1 1]
# [0 0 0 0 0 0 0 0 0 1 1 1]
# [0 0 0 0 0 0 0 1 1 1 1 1]
# [0 0 0 0 1 1 1 1 1 1 1 1]
# 对于上面的函数，第 2 行和第 6 行都有 8 个 1。所以输出[2,8] 和 [6,8]

import os

def traverseMatrix(matrix):
    # 时间复杂度：O（M+N）
    # 空间复杂度：O（1）
    if not matrix:
        return []
    m, n = len(matrix), len(matrix[0])
    index, num = 0, 0
    res = []
    i, j = m-1, n-1
    while i >= 0:
        while j >= 0:
            if matrix[i][j] == 1:
                j -= 1
            else:
                if matrix[i][j+1] == 1:
                    if (n-j-1) > num:
                        res = []
                        num = n - j - 1
                        index = i + 1
                        res.append([index, num])
                    elif (n-j-1) == num:
                        index = i + 1
                        res.append([index, num])
                break
        i -= 1
    return res

if __name__ == "__main__":
    # Matrix = [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    #           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    #           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    #           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    #           [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    #           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]
    # res= traverseMatrix(Matrix)
    # print(res)
    dir = os.path.join('/Users/maciel/Documents/GitProject/OpenNRE/data', 'nyt')
    print(dir)






