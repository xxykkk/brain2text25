import numpy as np
import sys, time
import json


def pagerank(d, M, R0, n):
    eps = 1e-8
    R = R0.copy()
    
    for i in range(1000):
        R_new = d * M @ R + (1-d) / n # 转移矩阵
        
        if np.linalg.norm(R_new - R, ord=1) < eps:
            return R_new, i
        R = R_new

    return R_new, 1000


if __name__ == '__main__':
    M = np.array(
        [[0, 1/2, 0, 0],
        [1/3, 0, 0, 1/2],
        [1/3, 0, 1, 1/2],
        [1/3, 1/2, 0, 0]
    ])

    N = 4
    d = 0.8 


    data = json.loads(input())
    M = np.array(data['edges'])
    N = data['N']
    R0 = np.ones(N) / N

    R = pagerank(d, M, R0, N)

    print(R)
    
    
    



