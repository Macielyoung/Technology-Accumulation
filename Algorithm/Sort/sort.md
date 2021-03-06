## 排序算法

![](/Users/maciel/Documents/Technology-Accumulation/Algorithm/Sort/sort.png)

1. 插入排序

   算法描述：每次都把一个数插入到前面已经排好序到列表中，直至列表中最后一个元素插入完成。

   时间复杂度：最好 O(n)   最坏 O(n^2)  平均 O(n^2)

   空间复杂度：O(1)

2. 选择排序

   算法描述：每次从待排序的列表中选择最小的（最大的）的数存放在列表的起始位置，直到所有元素都排序完成。

   时间复杂度：最好 O(n^2)   最坏 O(n^2)  平均 O(n^2)

   空间复杂度：O(1)

3. 冒泡排序

   算法描述：每次比较两个元素，如果大小顺序错误，则交换这两个数。

   时间复杂度：最好 O(n)   最坏 O(n^2)  平均 O(n^2)

   空间复杂度：O(1)

4. 快速排序

   算法描述：每次都将数据划分成两部分，左半部分都比pivot小，右半部分都比pivot大，然后对左右两半继续执行同样的操作，直至排序完成。

   时间复杂度：最好 O(nlogn)   最坏 O(nlogn)  平均 O(n^2)

   空间复杂度：O(1)

5. 希尔排序

   算法描述：把数组按下标的一定增量分组，对每组数据使用直接插入排序，随着增量变小，直至增量为1，算法终止。（增量一般取半来变化）

   时间复杂度：最好 O(nlogn)   最坏 O(nlogn)  平均 O(nlogn)

   空间复杂度：O(1)

6. 基数排序

   算法描述：数排序是按照低位先排序，再往高位排序，直至最高位。基数排序是基于分别排序，所以是稳定的。

   时间复杂度：O(d(n+r)) r为基数d为位数

   空间复杂度：O(n+rd)

7. 堆排序

   算法描述：利用堆积树这种数据结构设计的排序算法。堆分为大根堆和小根堆，是完全二叉树。大根堆的要求是每个节点的值都不大于其父节点的值，即A[PARENT[i]] >= A[i]。

   堆排序基本思路：

   （1）构造初始堆。将无序序列构造成一个堆（一般升序用大顶堆，降序用小顶堆）。

   （2）将堆顶元素与末尾元素进行交换，使末尾元素最大。然后前面的序列重新构造成最大（小）堆结构。如此重复建堆，交换操作。

   时间复杂度：最好 O(n)   最坏 O(n^2)  平均 O(n^2)

   空间复杂度：O(1)

8. 归并排序

   算法描述：将已有顺序的子列表合并，得到完全有序的列表。若将两个有序表合并成一个有序表，称为二路归并。

   时间复杂度：最好 O(n)  最坏 O(nlogn)  平均 O(nlogn)

   空间复杂度：O(1)