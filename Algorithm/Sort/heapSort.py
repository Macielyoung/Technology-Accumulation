#-*- coding: UTF-8 -*-

class Solution(object):
    def shift(self, array, start, end):
        # 调整成为最大堆
        lchild = start * 2 + 1
        rchild = start * 2 + 1
        maximum = start
        if maximum < end // 2:
            if array[lchild] > array[maximum]:
                maximum = lchild
            # 判断是否有右孩子
            if rchild < end:
                if array[rchild] > array[maximum]:
                    maximum = rchild
            if maximum != start:
                array[start], array[maximum] = array[maximum], array[start]
                self.shift(array, maximum, end)  # 递归

    def heap_sort(self, array):
        # 堆排序
        end = len(array)
        # 创建最大堆
        for i in range(end // 2 - 1, -1, -1):
            self.shift(array, i, end)
        for i in range(1, end):
            array[0], array[end - i] = array[end - i], array[0]
            self.shift(array, 0, end - i)
        return array

if __name__ == '__main__':
    solu = Solution()
    array = [16, 7, 3, 20, 17, 8]
    res = solu.heap_sort(array)
    print(res)