import numpy as np

def ldl_t_decompose(a):
    n = a.shape[0]
    b = a.copy()
    for k in range(n):
        w = 1/b[k,k]
        for i in range(k+1, n):
            b[i, k] = b[i, k]*w
        for j in range(k+1, n):
            v = b[k, k]*b[j, k]
            for i in range(j, n):
                b[i, j] = b[i, j] - b[i, k]*v
    l = np.identity(n)
    d = np.zeros_like(a)
    for i in range(n):
        d[i, i] = b[i, i]
        for j in range(i):
            if i > j:
                l[i, j] = b[i, j]
    return l, d


def ldlt_decompose_for_tridiagonal_matrix(a):
    n = a.shape[0]
    b = a.copy()
    for k in range(n - 1):
        w = 1/b[k,k]
        kp1 = k + 1
        b[kp1, k] = b[kp1, k]*w
        v = b[k, k]*b[kp1, k]
        b[kp1, kp1] = b[kp1, kp1] - b[kp1, k]*v

    l = np.identity(n)
    d = np.zeros_like(a)
    for i in range(n - 1):
        d[i, i] = b[i, i]
        l[i + 1, i] = b[i + 1, i]
    d[n - 1, n - 1] = b[n - 1, n - 1]
    return l, d
