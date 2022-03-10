import numpy as np


def lu_decompose_v1(a):
    n = a.shape[0]
    l = np.identity(n)
    u = a.copy()
    for k in range(n):
        for i in range(k + 1, n):
            l[i, k] = u[i, k]/u[k, k]
            u[i, k] = 0
            for j in range(k + 1, n):
                u[i,j] = u[i,j] - l[i, k] * u[k, j]
    return l, u


def lu_decompose_v2(a):
    n = a.shape[0]
    t = a.copy()
    for k in range(n):
        w = 1/t[k, k]
        for i in range(k + 1, n):
            t[i, k] = t[i, k]*w
            for j in range(k + 1, n):
                t[i,j] = t[i,j] - t[i, k] * t[k, j]
    l = np.identity(n)
    u = np.zeros_like(a)
    for i in range(n):
        for j in range(n):
            if i > j:
                l[i, j] = t[i, j]
            else:
                u[i, j] = t[i, j]
    return l, u

lu_decompose = lu_decompose_v2