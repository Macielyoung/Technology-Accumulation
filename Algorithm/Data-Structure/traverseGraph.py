#!/usr/bin/env python
# coding:utf-8

# @Created : Macielyoung
# @Time : 2018/11/26
# @Function : Traverse graph（DFS & BFS)
# Use adjacency list to store the graph instead of adjacent matrix which occupy much memory

# 邻接表：  对于图中的每个点，将它的邻居放到一个链表中
# 邻接矩阵：对于n个点，构造一个n*n的矩阵，如果有从点i到j的边，就将矩阵的位置matrix[i][j]置1

class Travese(object):
    def __init__(self):
        self.bfs_visited = []
        self.dfs_visited = []
        self.rdfs_searched = set()
        self.rdfs_res = []

    # 迭代法BFS
    def Breadth_First_Search_Iterative(self, graph, start):
        search_list = []
        result = []
        search_list.append(start)
        while search_list:
            cur = search_list.pop(0)
            if cur not in self.bfs_visited:
                # print(cur)
                result.append(cur)
                self.bfs_visited.append(cur)
                for node in graph[cur]:
                    if node not in search_list:
                        search_list.append(node)
        return result

    # 迭代法DFS
    def Depth_First_Search_Iterative(self, graph, start):
        idfs_searched = []
        result = []
        idfs_searched.append(start)
        while idfs_searched:
            cur = idfs_searched.pop()
            if cur not in self.dfs_visited:
                result.append(cur)
                self.dfs_visited.append(cur)
                for node in graph[cur][::-1]:
                    idfs_searched.append(node)
        return result

    # 递归法DFS
    def Depth_First_Search_Recursive(self, graph, start):
        if start not in self.rdfs_searched:
            self.rdfs_res.append(start)
            self.rdfs_searched.add(start)
        for node in graph[start]:
            if node not in self.rdfs_searched:
                self.Depth_First_Search_Recursive(graph, node)
        return self.rdfs_res

if __name__ == '__main__':
    GRAPH = {
        'A': ['B', 'F'],
        'B': ['C', 'I', 'G'],
        'C': ['B', 'I', 'D'],
        'D': ['C', 'I', 'G', 'H', 'E'],
        'E': ['D', 'H', 'F'],
        'F': ['A', 'G', 'E'],
        'G': ['B', 'F', 'H', 'D'],
        'H': ['G', 'D', 'E'],
        'I': ['B', 'C', 'D']
    }

    traverse = Travese()

    print('BFS Result:')
    bfs_res = traverse.Breadth_First_Search_Iterative(GRAPH, 'A')
    print(bfs_res)

    print('Iterative DFS Result:')
    idfs_res = traverse.Depth_First_Search_Iterative(GRAPH, 'A')
    print(idfs_res)

    print('Recursive DFS Result:')
    rdfs_res = traverse.Depth_First_Search_Recursive(GRAPH, 'A')
    print(rdfs_res)