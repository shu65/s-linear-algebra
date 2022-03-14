import numpy as np

from sla.qr_decompose import qr_decompose_modified_gram_schmidt


def shift_qr_method(a, s, max_iterations=100, verbose=False):
    identity_n = np.identity(a.shape[0])
    s_matrix = s * identity_n
    a_k = a
    for i in range(max_iterations):
        q_k, r_k = qr_decompose_modified_gram_schmidt(a_k - s_matrix)
        a_k = r_k @ q_k + s_matrix
        if verbose:
            eigen_values = np.diag(a_k)
            print("iteration", i, "eigen_values", eigen_values)
    eigen_values = np.diag(a_k)
    return eigen_values