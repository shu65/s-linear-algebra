import numpy as np

from sla.cholesky_decompose import cholesky_decompose


def cholesky_lr_method(a, max_iterations=100, verbose=False):
    a_k = a
    for i in range(max_iterations):
        r_k = cholesky_decompose(a_k)
        a_k = r_k.T @ r_k
        if verbose:
            eigen_values = np.diag(a_k)
            print("iteration", i, "eigen_values", eigen_values)
    eigen_values = np.diag(a_k)
    return eigen_values