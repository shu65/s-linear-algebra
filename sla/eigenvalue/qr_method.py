import numpy as np

from sla.qr_decompose import qr_decompose_modified_gram_schmidt


def qr_method(a, max_iterations=100, verbose=False):
    n = a.shape[0]
    a_k = a
    eigen_vectors = np.identity(n)
    for i in range(max_iterations):
        q_k, r_k = qr_decompose_modified_gram_schmidt(a_k)
        a_k = r_k @ q_k
        eigen_vectors = np.dot(eigen_vectors, q_k)
        if verbose:
            eigen_values = np.diag(a_k)
            print("iteration", i, "eigen_values", eigen_values)
    eigen_values = np.diag(a_k)
    return eigen_values, eigen_vectors