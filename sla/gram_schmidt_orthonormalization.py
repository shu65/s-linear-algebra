import numpy as np


def gram_schmidt_orthonormalization(a):
    n = a.shape[0]
    m = a.shape[1]
    assert n >= m
    q = np.zeros((n, m))
    for k in range(m):
        b = a[:, k]
        for j in range(k):
            b = b - q[:, j] * (q[:, j].T @ a[:, k])
        q[:, k] = b / np.linalg.norm(b)
    return q