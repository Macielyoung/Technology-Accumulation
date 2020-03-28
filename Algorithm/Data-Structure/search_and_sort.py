# 剑指offer中常见算法题
import queue

# 在二维数组中查找一个数是否存在 （Cha2.3）
# 该数组满足每一行自左向右递增，自上而下递增的特性
def FindInPartiallySortedList(nums: list, value: int):
    m, n = len(nums), len(nums[0])
    if m == 0 or n == 0:
        return False
    i, j = 0, n-1
    while i < m and j>= 0:
        if nums[i][j] == value:
            return True
        elif nums[i][j] < value:
            i += 1
        else:
            j -= 1
    # print(i, j)
    return False

# 树结构，常见的遍历算法（前序遍历、中序遍历、后序遍历、宽度优先遍历、深度优先遍历等）
# 已知前序遍历和中序遍历结果，构建原始二叉树
class TreeNode(object):
    def __init__(self, value):
        self.left = None
        self.right = None
        self.data = value

def construct_from_preorder_and_inorder(pre_list, in_list):
    if len(pre_list) == 0 or len(in_list) == 0:
        return None
    root = TreeNode(pre_list[0])
    left_pre_list, right_pre_list, left_in_list, right_in_list = find_sublist(pre_list, in_list, pre_list[0])
    # 递归式构建左子树和右子树
    root.left = construct_from_preorder_and_inorder(left_pre_list, left_in_list)
    root.right = construct_from_preorder_and_inorder(right_pre_list, right_in_list)
    return root

def find_sublist(pre_list, in_list, value):
    in_loc = 0
    for i in range(len(in_list)):
        if in_list[i] == value:
            in_loc = i
            break
    left_in_list = in_list[:in_loc]
    right_in_list = in_list[in_loc+1:]

    left_num = len(left_in_list)
    left_pre_list = pre_list[1:1+left_num]
    right_pre_list = pre_list[1+left_num:]
    return left_pre_list, right_pre_list, left_in_list, right_in_list

# 宽度优先遍历
def bfs(root):
    node_data = []
    if root == None:
        return node_data
    q = queue.Queue()
    q.put(root)
    while not q.empty():
        node = q.get()
        node_data.append(node.data)
        if node.left != None:
            q.put(node.left)
        if node.right != None:
            q.put(node.right)
    return node_data

# 深度优先遍历（前序遍历、中序遍历和后序遍历）
# (1)前序遍历 - 递归和迭代两种
def preorder_recursive(root, node_data):
    if root == None:
        return None
    node_data.append(root.data)
    preorder_recursive(root.left, node_data)
    preorder_recursive(root.right, node_data)

def preorder_iterative(root):
    node_data = []
    if root == None:
        return node_data
    stack = [root]
    while stack:
        node = stack.pop()
        node_data.append(node.data)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return node_data

# (2)中序遍历 - 递归和迭代两种
def inorder_recursive(root, node_data):
    if root == None:
        return None
    inorder_recursive(root.left, node_data)
    node_data.append(root.data)
    inorder_recursive(root.right, node_data)

def inorder_iterative(root):
    node_data = []
    if root == None:
        return node_data
    stack = []
    cur = root
    while cur != None or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        if stack:
            cur = stack.pop()
            node_data.append(cur.data)
            cur = cur.right
    return node_data

# (3)后序遍历 - 递归和迭代两种
def postorder_recursive(root, node_data):
    if root == None:
        return None
    postorder_recursive(root.left, node_data)
    postorder_recursive(root.right, node_data)
    node_data.append(root.data)

# 左右中是中右左的逆序结果
def postorder_iterative(root):
    node_data = []
    if root == None:
        return node_data
    stack = [root]
    while stack:
        cur = stack.pop()
        node_data.append(cur.data)
        if cur.left:
            stack.append(cur.left)
        if cur.right:
            stack.append(cur.right)
    return node_data[::-1]

# 栈和队列
        
if __name__ == "__main__":
    # # 1. 二维数组查找
    # nums = [[1,2,8,9],
    #         [2,4,9,12],
    #         [4,7,10,13],
    #         [6,8,11,15]]
    # value = 5
    # res = FindInPartiallySortedList(nums, value)
    # print(res)

    # # 2. 构建二叉树，并输出
    # pre_list = [1, 2, 4, 7, 3, 5, 6, 8]
    # in_list = [4, 7, 2, 1, 5, 3, 8 ,6]
    # root = construct_from_preorder_and_inorder(pre_list, in_list)
    # # node_data = bfs(root)
    # # print(node_data)
    # node_data = []
    # preorder_recursive(root, node_data)
    # print(node_data)
    # node_data2 = preorder_iterative(root)
    # print(node_data2)

    # node_data3 = []
    # inorder_recursive(root, node_data3)
    # print(node_data3)
    # node_data4 = inorder_iterative(root)
    # print(node_data4)

    # node_data5 = []
    # postorder_recursive(root, node_data5)
    # print(node_data5)
    # node_data6 = postorder_iterative(root)
    # print(node_data6)