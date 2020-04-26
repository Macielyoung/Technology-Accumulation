import numpy as np

# 递归、迭代、动归问题

def fibonacci_recursive(n):
    # 时间复杂度是n的指数次
    if n == 0 or n == 1:
        return n
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_iterative(n):
    # 时间复杂度是O(n)
    if n == 0 or n == 1:
        return n
    min_num = 0
    max_num = 1
    for _ in range(2, n+1):
        tmp = max_num
        max_num += min_num
        min_num = tmp
    return max_num

# 利用矩阵公式
# f(n),   f(n-1) = 1  1 ^ (n-1)
# f(n-1), f(n-2) = 1  0
# a^n = a^(n/2) * a^(n/2)  n为偶数
# a^n = a^((n-1)/2) * a^((n-1)/2) * a  n为奇数
# 利用递归的方式计算，时间复杂度为O(logn)
def cal_matrix(n, res):
    # res = np.mat[[1, 1], [1, 0]]
    if n == 2:
        return res
    if n == 3:
        return res * res
    if n % 2 == 1:
        sqrt_res = cal_matrix((n-1)/2, res)
        print(sqrt_res)
        return sqrt_res * sqrt_res * res
    if n % 2 == 0:
        sqrt_res = cal_matrix(n/2, res)
        return sqrt_res * sqrt_res

def fibonacci_formula(n, mat):
    if n == 0 or n == 1:
        return n
    mat_res = cal_matrix(n, mat)
    print(mat_res)
    res = np.array(mat_res)[0][0]
    return res



if __name__ == '__main__':
    n = 5
    res1 = fibonacci_recursive(n)
    print("res1: ", res1)
    res2 = fibonacci_iterative(n)
    print("res2: ", res2)

    mat = np.mat([[1, 1], [1, 0]])
    mat_res = fibonacci_formula(n+1, mat)
    print(mat_res)
