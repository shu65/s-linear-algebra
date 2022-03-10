import numpy as np


def qr_decompose_gram_schmidt(a):
    n = a.shape[0]
    m = a.shape[1]
    assert n >= m
    q = np.zeros((n, m))
    r = np.zeros((m, m))
    for k in range(m):
        b = a[:, k]
        for j in range(k):
            r[j, k] = q[:, j].T @ a[:, k]
            b = b - q[:, j] * r[j, k]
        r[k, k] = np.linalg.norm(b)
        q[:, k] = b / r[k, k]

    return q, r


def qr_decompose_modified_gram_schmidt(a):
    n = a.shape[0]
    m = a.shape[1]
    assert n >= m
    u = np.zeros((n, m))
    q = np.zeros((n, m))
    r = np.zeros((m, m))
    for j in range(m):
        u[:, j] = u[:, j] + a[:, j]
        q[:, j] = u[:, j]/np.linalg.norm(u[:, j])
        r[j,j] = np.linalg.norm(u[:, j])
        for k in range(j + 1, m):
            r[j ,k] = q[:, j].T @ a[:, k]
            u[:, k] = u[:, k] - q[:, j] * r[j, k]
    return q, r
