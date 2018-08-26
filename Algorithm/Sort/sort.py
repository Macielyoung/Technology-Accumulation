#-*- coding: UTF-8 -*-

# Author : Macielyoung
# Time : 2018/8/26
# Function : Sort algorithms with Python

# 插入排序
def insert_sort(nums):
    # 时间复杂度 ： 最好 O(n)  最坏 O(n^2)  平均 O(n^2)
    # 空间复杂度 ： O(1)
    # 稳定性 : 稳定
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] < nums[j]:
                # 将待插入待数放到对应的位置，后面的数依次向后移动
                nums.insert(j, nums.pop(i))
                break
    return nums

# 选择排序
def select_sort(nums):
    # 时间复杂度 ： 最好 O(n^2)  最坏 O(n^2)  平均 O(n^2)
    # 空间复杂度 ： O(1)
    # 稳定性 : 不稳定
    for i in range(len(nums)):
        x = i
        for j in range(i, len(nums)):
            if nums[j] < nums[x]:
                x = j
        nums[i], nums[x] = nums[x], nums[i]
    return nums

# 冒泡排序
def bubble_sort(nums):
    # 时间复杂度 ： 最好 O(n)  最坏 O(n^2)  平均 O(n^2)
    # 空间复杂度 ： O(1)
    # 稳定性 : 稳定
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            if(nums[i] > nums[j]):
                nums[i], nums[j] = nums[j], nums[i]
    return nums

# 快速排序
def quick_sort(nums):
    # 时间复杂度 ： 最好 O(nlogn)  最坏 O(nlogn)  平均 O(n^2)
    # 空间复杂度 ： O(1)
    # 稳定性 : 不稳定
    def partition(begin, end):
        if begin > end:
            return
        l, r = begin, end
        pivot = nums[l]
        while l < r:
            while l < r and nums[r] > pivot:
                r -= 1
            while l < r and nums[l] <= pivot:
                l += 1
            nums[l], nums[r] = nums[r], nums[l]
        nums[l], nums[begin] = nums[begin], nums[l]
        partition(begin, l-1)
        partition(r+1, end)
    partition(0, len(nums)-1)
    return nums

# 希尔排序
def shell_sort(nums):
    # 时间复杂度 ： 最好 O(nlogn)  最坏 O(nlogn)  平均 O(nlogn)
    # 空间复杂度 ： O(1)
    # 稳定性 : 不稳定
    gap = len(nums)
    while gap > 1:
        gap = gap // 2
        for i in range(gap, len(nums)):
            for j in range(i % gap, i, gap):
                if nums[i] < nums[j]:
                    nums[i], nums[j] = nums[j], nums[i]
    return nums

# 基数排序
def radix_sort(nums):
    # 时间复杂度 ： O(d(n+r)) r为基数d为位数
    # 空间复杂度 ： O(n+rd)
    # 稳定性 : 稳定
    bucket, digit = [[]], 0
    while len(bucket[0]) != len(nums):
        bucket = [[], [], [], [], [], [], [], [], [], []]
        for i in range(len(nums)):
            num = (nums[i] // 10 ** digit) % 10
            bucket[num].append(nums[i])
        del nums[:]
        for i in range(len(bucket)):
            nums += bucket[i]
        digit += 1
    return nums

# 堆排序
def max_heapify(nums, i, size):
    # 最大堆调整
    # 注意数组的size比数组的最大索引大1
    lchild = 2 * i + 1
    rchild = 2 * i + 2
    maximum = i
    if i < size // 2:
        if nums[lchild] > nums[maximum]:
            maximum = lchild
        if rchild < size:
            # 肯定有左子节点，未必有右子节点
            if nums[rchild] > nums[maximum]:
                maximum = rchild
        if maximum != i:
            nums[i], nums[maximum] = nums[maximum], nums[i]
            max_heapify(nums, maximum, size)  # 递归
def build_max_heap(nums, size):
    # 创建最大堆
    for i in range(size // 2 - 1, -1, -1):
        max_heapify(nums, i, size)
def heap_sort(nums):
    # 时间复杂度 ： 最好 O(nlogn)  最坏 O(nlogn)  平均 O(nlogn)
    # 空间复杂度 ： O(1)
    # 稳定性 : 不稳定
    size = len(nums)
    build_max_heap(nums, size)
    for i in range(1, size):
        nums[0], nums[size - i] = nums[size - i], nums[0]
        max_heapify(nums, 0, size - i)
    return nums

# 归并排序
def merge_sort(nums):
    # 时间复杂度 ： 最好 O(n)  最坏 O(nlogn)  平均 O(nlogn)
    # 空间复杂度 ： O(1)
    # 稳定性 : 稳定
    def merge_arr(arr_l, arr_r):
        array = []
        while len(arr_l) and len(arr_r):
            if arr_l[0] <= arr_r[0]:
                array.append(arr_l.pop(0))
            elif arr_l[0] > arr_r[0]:
                array.append(arr_r.pop(0))
        if len(arr_l) != 0:
            array += arr_l
        elif len(arr_r) != 0:
            array += arr_r
        return array

    def recursive(nums):
        if len(nums) == 1:
            return nums
        mid = len(nums) // 2
        arr_l = recursive(nums[:mid])
        arr_r = recursive(nums[mid:])
        return merge_arr(arr_l, arr_r)

    return recursive(nums)

if __name__ == "__main__":
    numbers = [1, 3, 4, 1, 0, 29, 203, 213, 294]

    insert_res = insert_sort(numbers)
    print(insert_res)

    select_res = select_sort(numbers)
    print(select_res)

    bubble_res = bubble_sort(numbers)
    print(bubble_res)

    quick_res = quick_sort(numbers)
    print(quick_res)

    shell_res = shell_sort(numbers)
    print(shell_res)

    radix_res = radix_sort(numbers)
    print(radix_res)

    heap_res = heap_sort(numbers)
    print(heap_res)

    merge_res = merge_sort(numbers)
    print(merge_res)







