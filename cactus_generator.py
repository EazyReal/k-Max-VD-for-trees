import random
import os
import numpy as np

class RandomCatusGenerator:
    # find a seed graph G
    # then find a spanning tree T
    # use T on G extend to a cactus by tries t
    # https://link.springer.com/chapter/10.1007/978-3-030-10448-1_12
    def __init__(self):
        pass
    
    @staticmethod
    def findG(n, p):
        G = np.random.binomial(1, p, n*n).reshape(n, n)
        # self loop ok 
        return G
    
    @staticmethod
    def findT(n, G):
        vis = [False]*n
        T = np.zeros((n,n))
        def dfs(u):
            for v in range(n):
                if(G[u][v] != 0 and not vis[v]):
                    vis[v] = 1
                    T[u][v] = T[v][u] = 1
                    dfs(v)
                    
        for i in range(n):
            if(not vis[i]):
                vis[i] = 1
                if(i > 0):
                    T[i-1][i] = T[i][i-1] = 1
                dfs(i)
                
        ## assert is T
        deg = 0
        for i in range(n):
            for j in range(n):
                if(i == j):
                    assert(T[i][j] == 0)
                deg += T[i][j]
        assert(deg == 2*n-2)
        return T
    
    @staticmethod
    def findK(n, T, G, t):
        def conflict(x, y):
            return 1
            pass
        try_time = 0
        K = T
        while(try_time < t):
            u = random.randint(1, n)
            v = random.randint(1, n)
            if(u == v or conflict(u, v)):
                try_time += 1
                continue
            else:
                K[u][v] = K[v][u] = 1
        return K
        
    @staticmethod
    # n = #vertex
    # p = prob of connection to find 
    def generate(n, p, t):
        G = RandomCatusGenerator.findG(n,p)
        T = RandomCatusGenerator.findT(n, G)
        K = RandomCatusGenerator.findK(n, T, G, t)
        return K
    
    def generate_tree(n, p):
        G = RandomCatusGenerator.findG(n,p)
        T = RandomCatusGenerator.findT(n, G)
        return T
    
   ## end of class