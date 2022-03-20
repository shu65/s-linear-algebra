import numpy as np


def convert_tridiagonal(a):
    m = a.shape[0]
    n = a.shape[1]
    h = a.copy()
    q = np.identity(m)
    for k in range(n - 2):
        v_k = h[k + 1:, [k]].copy()
        v_k[0] = v_k[0] + np.sign(v_k[0]) * np.linalg.norm(v_k)
        v_k = v_k / np.linalg.norm(v_k)
        h[k + 1:, k:] = h[k + 1:, k:] - 2 * v_k @ (v_k.T @ h[k + 1:, k:])
        h[:, k + 1:] = h[:, k + 1:] - 2 * h[:, k + 1:] @ v_k @ v_k.T
        q[k + 1:, k + 1:] = q[k + 1:, k + 1:] - 2 * v_k @ (v_k.T @ q[k + 1:, k + 1:])
    return q, h
