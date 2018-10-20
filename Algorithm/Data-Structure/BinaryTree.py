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


if __name__ == '__main__':
    traverse = TraverseBT()
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
    # 前序遍历
    # res = traverse.recursive_preOrder(A)
    # res2 = traverse.iterative_preOrder(A)
    # print(res, res2)
    # 中序遍历
    # res3 = traverse.recursive_inOrder(A)
    # res4 = traverse.iterative_inOrder(A)
    # print(res3, res4)
    # 后序遍历
    # res5 = traverse.recursive_postOrder(A)
    # res6 = traverse.iterative_postOrder(A)
    # print(res5, res6)
    # 层次遍历
    res7 = traverse.levelOrder(A)
    print(res7)
