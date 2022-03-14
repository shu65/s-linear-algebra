import numpy as np


def is_upper_triangular_matrix(a):
    return np.array_equal(a, np.triu(a))


def is_lower_triangular_matrix(a):
    return np.array_equal(a, np.tril(a))


def is_orthogonal_matrix(a):
    return np.allclose(a.T @ a, np.identity(a.shape[1]))


def is_symmetric(a):
    return np.array_equal(a, a.T)


def is_hermitian(a):
    return np.array_equal(a, np.conjugate(a))

def is_positive_definite(a):
    if not is_symmetric(a):
        raise ValueError("unsupported no symmetric")
    return np.linalg.det(a)