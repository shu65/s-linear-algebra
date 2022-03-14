import numpy as np

from sla.eigenvalue.simultaneous_iteration import simultaneous_iteration


def arnoldi_method_v1(a, b, m):
    n = a.shape[0]
    h = np.zeros((m, m))
    v = np.zeros((n, m))

    q = b / np.linalg.norm(b)
    v[:, 0] = q

    for k in range(m):
        w = a @ v[:, k]
        for j in range(k+1):
            v_j = v[:, j]
            h[j,k] = v_j.T @ w
            w = w - h[j,k]*v_j

        if k+1 < m:
            h[k+1, k] = np.linalg.norm(w)
            v[:, k+1] = w/h[k+1, k]
    return h, v

arnoldi_method = arnoldi_method_v1

def simultaneous_iteration_arnoldi(a, x0, m, max_iterations=100, verbose=False):
    h, v = arnoldi_method(a, x0, m)
    eigen_values, y = simultaneous_iteration(h, max_iterations=max_iterations, verbose=verbose)
    eigen_vectors = v @ y
    return eigen_values, eigen_vectors
