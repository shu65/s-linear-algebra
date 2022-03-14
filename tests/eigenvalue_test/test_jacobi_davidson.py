import numpy as np

from sla.eigenvalue.jacobi_davidson import jacobi_davidson
from sla.utility import is_orthogonal_matrix
from sla.gram_schmidt_orthonormalization import gram_schmidt_orthonormalization


def test_simultaneous_iteration_lanczos():
    np.random.seed(40)
    n = 6
    expected_eigen_values = 1/np.linspace(1., n, n)
    expected_eigen_vectors = gram_schmidt_orthonormalization(np.random.randn(n, n))
    a_matrix = np.linalg.solve(expected_eigen_vectors, np.diag(expected_eigen_values) @ expected_eigen_vectors)
    x0 = np.random.randn(n)
    n_eigen_values = 2
    max_k = 2*n_eigen_values
    eps = 1e-3
    eigen_values, eigen_vectors = jacobi_davidson(a_matrix, x0=x0, max_k=max_k, eps=eps, verbose=True)
    #assert len(eigen_values) == m
    assert is_orthogonal_matrix(eigen_vectors)
    np.testing.assert_almost_equal(
        a_matrix @ eigen_vectors,
        eigen_vectors @ np.diag(eigen_values),
        decimal=1
    )