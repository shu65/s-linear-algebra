import numpy as np

from sla.eigenvalue.simultaneous_iteration import simultaneous_iteration
from sla.utility import is_hermitian

def lanczos_method_v1(a, x0, m):
    assert is_hermitian(a)

    n = a.shape[0]
    alpha = np.zeros(m)
    beta = np.zeros(m)
    v = np.zeros((n, m))

    v0 = x0 / np.linalg.norm(x0)
    v[:, 0] = v0
    v_k_m1 = np.zeros(n)
    for k in range(m - 1):
        v_k = v[:, k]
        w = a @ v_k - beta[k] * v_k_m1
        alpha[k] = w @ v_k
        w = w - alpha[k] * v_k
        beta[k + 1] = np.sqrt(w @ w)
        v[:, k + 1] = w / beta[k + 1]
        v_k_m1 = v_k
    t = np.diag(alpha)
    for i in range(0, m-1):
        t[i, i+1] = beta[i+1]
        t[i+1, i] = beta[i+1]
    return t, v

lanczos_method = lanczos_method_v1

def simultaneous_iteration_lanczos(a, n_eigen_values, x0, m, max_iterations=100, verbose=False):
    assert n_eigen_values <= m
    t, v = lanczos_method(a, x0, m)
    #eigen_values, y = simultaneous_iteration(h, max_iterations=max_iterations, verbose=verbose)
    eigen_values, y = np.linalg.eig(t)
    eigen_vectors = v @ y
    return eigen_values[:n_eigen_values], eigen_vectors[:, :n_eigen_values]
