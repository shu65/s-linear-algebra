import numpy as np


def is_upper_triangular_matrix(a):
    return np.array_equal(a, np.triu(a))


def is_lower_triangular_matrix(a):
    return np.array_equal(a, np.tril(a))


def is_orthogonal_matrix(a):
    return np.allclose(a.T @ a, np.identity(a.shape[1]))