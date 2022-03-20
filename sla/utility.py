import numpy as np


def is_upper_triangular_matrix(a):
    return np.array_equal(a, np.triu(a))


def is_lower_triangular_matrix(a):
    return np.array_equal(a, np.tril(a))


def is_orthogonal_matrix(a):
    print("is_orthogonal_matrix", a.T @ a)
    return np.allclose(a.T @ a, np.identity(a.shape[1]))


def is_symmetric(a):
    return np.array_equal(a, a.T)


def is_hermitian(a):
    return np.array_equal(a, np.conjugate(a))

def is_square_matrix(a):
    n = a.shape[0]
    m = a.shape[1]
    return n == m

def is_tridiagonal_matrix(a):
    if not is_square_matrix(a):
        return False
    expeccted_t = np.diag(np.diag(a))
    l = np.diag(a, k=-1)
    u = np.diag(a, k=1)
    for i in range(len(l)):
        expeccted_t[i, i + 1] = l[i]
        expeccted_t[i + 1, i] = u[i]
    return np.allclose(a, expeccted_t)

def is_positive_definite(a):
    if not is_symmetric(a):
        raise ValueError("unsupported no symmetric")
    return np.linalg.det(a)