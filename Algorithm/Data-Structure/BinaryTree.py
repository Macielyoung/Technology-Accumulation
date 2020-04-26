#-*- coding: UTF-8 -*-

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 二叉树的遍历
class TraverseBT(object):
    def __init__(self):
        self.pre_res = []
        self.in_res = []
        self.pos_res = []

    # 递归前序遍历二叉树
    def recursive_preOrder(self, root):
        if not root:
            return None
        self.pre_res.append(root.val)
        self.recursive_preOrder(root.left)
        self.recursive_preOrder(root.right)
        return self.pre_res

    # 非递归前序遍历二叉树
    def iterative_preOrder(self, root):
        if not root:
            return None
        res = []
        stack = [root]
        while stack:
            cur = stack.pop()
            res.append(cur.val)
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        return res

    # 递归中序遍历二叉树
    def recursive_inOrder(self, root):
        if not root:
            return None
        self.recursive_inOrder(root.left)
        self.in_res.append(root.val)
        self.recursive_inOrder(root.right)
        return self.in_res

    # 非递归中序遍历二叉树
    def iterative_inOrder(self, root):
        if not root:
            return None
        res = []
        stack = []
        cur = root
        while cur!=None or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            if stack:
                cur = stack.pop()
                res.append(cur.val)
                cur = cur.right
        return res

    # 递归后序遍历二叉树
    def recursive_postOrder(self, root):
        if not root:
            return None
        self.recursive_postOrder(root.left)
        self.recursive_postOrder(root.right)
        self.pos_res.append(root.val)
        return self.pos_res

    # 非递归后序遍历二叉树
    def iterative_postOrder(self, root):
        if not root:
            return None
        stack = [root]
        res = []
        while stack:
            cur = stack.pop()
            res.append(cur.val)
            if cur.left:
                stack.append(cur.left)
            if cur.right:
                stack.append(cur.right)
        return res[::-1]

    # 层次遍历二叉树，即广度优先遍历二叉树
    def levelOrder(self, root):
        if not root:
            return None
        stack = [root]
        res = []
        while stack:
            cur = stack.pop(0)
            res.append(cur.val)
            if cur.left:
                stack.append(cur.left)
            if cur.right:
                stack.append(cur.right)
        return res

# 统计二叉树节点个数
class CountBT(object):
    def count_nodes_bfs(self, root):
        if not root:
            return 0
        # 广度优先遍历二叉树
        stack = [root]
        res = 0
        while stack:
            cur = stack.pop(0)
            res += 1
            if cur.left:
                stack.append(cur.left)
            if cur.right:
                stack.append(cur.right)
        return res

    def count_nodes_dfs(self, root):
        if not root:
            return 0
        # 深度优先遍历二叉树(即中序遍历二叉树)
        stack = []
        cur = root
        res = 0
        while cur!=None or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            if stack:
                cur = stack.pop()
                res += 1
                cur = cur.right
        return res

    # 递归求出二叉树的节点数量
    def recursive_count(self, root):
        if not root:
            return 0
        return self.recursive_count(root.left)+self.recursive_count(root.right)+1

# 求二叉树的深度
class HighBT(object):
    # 递归求二叉树深度
    def recursive_high(self, root):
        if not root:
            return 0
        return max(self.recursive_high(root.left), self.recursive_high(root.right))+1

    # 非递归求二叉树深度
    def level_high(self, root):
        if not root:
            return 0
        stack = [(root,1)]
        height = 1
        while stack:
            cur, h = stack.pop()
            if h > height:
                height = h
            if cur.left:
                stack.append((cur.left, h+1))
            if cur.right:
                stack.append((cur.right, h+1))
        return height

# 求二叉树第k层的节点个数
class KLevelBT(object):
    def kLevel_node(self, root, k):
        if not root or k < 1:
            return 0
        res = 0
        stack = [(root, 1)]
        while stack:
            cur, h = stack.pop()
            if h==k:
                res += 1
            if cur.left:
                stack.append((cur.left, h+1))
            if cur.right:
                stack.append((cur.right, h+1))
        return res

    # 递归(返回以root节点为根的第k层节点数量等于root左孩子为根第k-1层和root右孩子为根第k-1层节点数量和)
    def recursive_klevel_node(self, root, k):
        if not root or k < 1:
            return 0
        if k == 1:
            return 1
        return self.recursive_klevel_node(root.left, k-1) + self.recursive_klevel_node(root.right, k-1)

# 计算二叉树叶子节点个数
class LeafBT(object):
    def count_leaf(self, root):
        if not root:
            return 0
        res = 0
        stack = [root]
        while stack:
            cur = stack.pop()
            if cur.left == None and cur.right == None:
                res += 1
            if cur.left:
                stack.append(cur.left)
            if cur.right:
                stack.append(cur.right)
        return res

    def recursive_count_leaf(self, root):
        if not root:
            return 0
        if root.left == None and root.right == None:
            return 1
        return self.recursive_count_leaf(root.left) + self.recursive_count_leaf(root.right)

# 判断两棵二叉树是否相同，判断一棵树是否是平衡二叉树,判断两棵树是否是镜像
class CompareBT(object):
    def isSame(self, root1, root2):
        if not root1 and not root2:
            return True
        elif root1 == None or root2 == None:
            return False
        if root1.val != root2.val:
            return False
        return self.isSame(root1.left, root2.left) and self.isSame(root1.right, root2.right)

    def level_high(self, root):
        if not root:
            return 0
        stack = [(root,1)]
        height = 1
        while stack:
            cur, h = stack.pop()
            if h > height:
                height = h
            if cur.left:
                stack.append((cur.left, h+1))
            if cur.right:
                stack.append((cur.right, h+1))
        return height

    def isAVL(self, root):
        if not root:
            return True
        if abs(self.level_high(root.left)-self.level_high(root.right)) > 1:
            return False
        return self.isAVL(root.left) and self.isAVL(root.right)

    def isMirror(self, root1, root2):
        if not root1 and not root2:
            return True
        elif not root1 or not root2:
            return False
        if (root1.val != root2.val):
            return False
        return self.isMirror(root1.left, root2.right) and self.isMirror(root1.right, root2.left)

# 翻转二叉树
class MirrorBT(object):
    def getMirror(self, root):
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.getMirror(root.left)
        self.getMirror(root.right)
        return root

    def invertTree(self, root):
        if not root:
            return None
        stack = [root]
        while stack:
            cur = stack.pop()
            cur.left, cur.right = cur.right, cur.left
            if cur.left:
                stack.append(cur.left)
            if cur.right:
                stack.append(cur.right)
        return root

# 求两个节点的最低公共祖先节点
class commonParentBT(object):
    def getCommonParent(self, root, n1, n2):
        if not root:
            return None
        if root==n1 or root==n2:
            return root
        commonLeft = self.getCommonParent(root.left, n1, n2)
        commonRight = self.getCommonParent(root.right, n1, n2)
        if commonLeft != None and commonRight != None:
            return root
        if commonLeft != None:
            return commonLeft
        return commonRight

# 判断一棵树是否是二叉搜索树
class BST(object):
    # 中序遍历应该是递增的
    def isBST(self, root):
        if not root:
            return True
        stack = []
        res = []
        cur = root
        while cur != None or stack:
            while cur:
                stack.append(cur)
                cur = cur.left
            if stack:
                cur = stack.pop()
                if len(res)!=0:
                    if cur.val < res[-1]:
                        return False
                res.append(cur.val)
                cur = cur.right
        return True

# 路径和问题（深度优先遍历）
class PathSum(object):
    # 是否存在路径和等于某个值
    def existPath(self, root, num):
        def traverse(node, num):
            if not node:
                return False
            num -= node.val
            if num == 0 and node.left == None and node.right == None:
                return True
            res = traverse(node.left, num) or traverse(node.right, num)
            return res
        res = traverse(root, num)
        return res

    # 返回所有路径，使之和等于某个值
    def getPath(self, root, num):
        res = []
        def traverse(node, num, path):
            if not node:
                return
            num -= node.val
            if num == 0 and node.left == None and node.right == None:
                path.append(node.val)
                res.append(path)
                return
            traverse(node.left, num, path + [node.val])
            traverse(node.right, num, path + [node.val])
        traverse(root, num, [])
        return res

if __name__ == '__main__':
    # 构建一棵二叉树
    A = TreeNode(1)
    B = TreeNode(2)
    C = TreeNode(3)
    D = TreeNode(4)
    E = TreeNode(5)
    F = TreeNode(6)
    G = TreeNode(7)
    A.left = B
    A.right = C
    B.left = D
    B.right = E
    C.left = F
    C.right = G

    # # 前序遍历
    # traverse = TraverseBT()
    # res = traverse.recursive_preOrder(A)
    # res2 = traverse.iterative_preOrder(A)
    # print(res, res2)
    # # 中序遍历
    # res3 = traverse.recursive_inOrder(A)
    # res4 = traverse.iterative_inOrder(A)
    # print(res3, res4)
    # # 后序遍历
    # res5 = traverse.recursive_postOrder(A)
    # res6 = traverse.iterative_postOrder(A)
    # print(res5, res6)
    # # 层次遍历
    # res7 = traverse.levelOrder(A)
    # print(res7)
    #
    # # 统计节点个数
    # count = CountBT()
    # nodes_num = count.count_nodes_bfs(A)
    # nodes_num2 = count.count_nodes_dfs(A)
    # nodes_num3 = count.recursive_count(A)
    # print(nodes_num, nodes_num2, nodes_num3)
    #
    # # 计算二叉树的高度
    # high = HighBT()
    # height = high.recursive_high(A)
    # height2 = high.level_high(A)
    # print(height, height2)
    #
    # # 计算二叉树每层的节点个数
    # klevel = KLevelBT()
    # klevelNum = klevel.kLevel_node(A, 3)
    # klevelNum2 = klevel.recursive_klevel_node(A, 3)
    # print(klevelNum, klevelNum2)
    #
    # # 计算二叉树叶子节点个数
    # leafs = LeafBT()
    # leaf_num = leafs.count_leaf(A)
    # leaf_num2 = leafs.recursive_count_leaf(A)
    # print(leaf_num, leaf_num2)
    #
    # # 判断两棵二叉树是否相同,一棵树是否是平衡二叉树
    # A2 = TreeNode(1)
    # B2 = TreeNode(2)
    # C2 = TreeNode(3)
    # D2 = TreeNode(4)
    # E2 = TreeNode(5)
    # F2 = TreeNode(6)
    # G2 = TreeNode(7)
    # A2.left = C2
    # A2.right = B2
    # B2.left = E2
    # B2.right = D2
    # C2.left = G2
    # C2.right = F2
    # comparation = CompareBT()
    # isSame = comparation.isSame(A, A2)
    # isAVL = comparation.isAVL(A)
    # isMirror = comparation.isMirror(A, A2)
    # print(isSame, isAVL, isMirror)
    #
    # 求一棵树的镜像(自上而下递归翻转)
    Mirror = MirrorBT()
    root = Mirror.getMirror(A)
    root2 = Mirror.invertTree(A)
    print(root.left.val, root.left.left.val)
    print(root2.left.val, root2.left.left.val)
    #
    # # 求两个节点的最近公共祖先
    # common = commonParentBT()
    # parents = common.getCommonParent(A, E, G)
    # print(parents.val)
    #
    # # 判断一棵树是否是二分搜索树
    # A3 = TreeNode(4)
    # B3 = TreeNode(2)
    # C3 = TreeNode(6)
    # D3 = TreeNode(1)
    # E3 = TreeNode(3)
    # F3 = TreeNode(5)
    # G3 = TreeNode(7)
    # A3.left = B3
    # A3.right = C3
    # B3.left = D3
    # B3.right = E3
    # C3.left = F3
    # C3.right = G3
    # bst = BST()
    # isBST = bst.isBST(A3)
    # print(isBST)

    # 路径和
    paths = PathSum()
    num = 8
    hasPath = paths.existPath(A, num)
    path = paths.getPath(A, num)
    print(hasPath, path)