import numpy as np

from sla.lu_decompose import lu_decompose


def lr_method(a, max_iterations=100, verbose=False):
    a_k = a
    for i in range(max_iterations):
        l_k, r_k = lu_decompose(a_k)
        a_k = r_k @ l_k
        if verbose:
            eigen_values = np.diag(a_k)
            print("iteration", i, "eigen_values", eigen_values)
    eigen_values = np.diag(a_k)
    return eigen_values