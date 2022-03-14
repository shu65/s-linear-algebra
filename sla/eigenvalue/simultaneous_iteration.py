import numpy as np

from sla.gram_schmidt_orthonormalization import gram_schmidt_orthonormalization
from sla.qr_decompose import qr_decompose_modified_gram_schmidt

def simultaneous_iteration(a, max_iterations=100, verbose=False):
    r = np.random.rand(a.shape[1], a.shape[1])
    eigen_vectors = gram_schmidt_orthonormalization(r)
    if verbose:
        print("initialized eigen_vectors", eigen_vectors)
    for i in range(max_iterations):
        y = a @ eigen_vectors
        eigen_vectors, r = qr_decompose_modified_gram_schmidt(y)
        if verbose:
            eigen_values = np.diag((eigen_vectors.T @ a) @ eigen_vectors)
            print("iteration", i)
            print("eigen_values", eigen_values)
            print("eigen_vectors", eigen_vectors)
            print()
    eigen_values = np.diag((eigen_vectors.T @ a) @ eigen_vectors)
    return eigen_values, eigen_vectors


