import numpy as np


def cholesky_decompose(a):
    n = a.shape[0]
    c = a.copy()
    for k in range(n):
        c[k, k] = np.sqrt(c[k, k])
        for j in range(k+1, n):
            c[k, j] = 0
        w = 1 / c[k, k]
        for i in range(k+1, n):
            c[i, k] = c[i, k] * w
        for j in range(k+1, n):
            for i in range(j, n):
                c[i, j] = c[i, j] - c[i, k]*c[j, k]
    return c