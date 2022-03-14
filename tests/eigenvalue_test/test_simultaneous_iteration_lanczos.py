import numpy as np

from sla.eigenvalue.simultaneous_iteration_lanczos import simultaneous_iteration_lanczos
from sla.utility import is_orthogonal_matrix
from sla.gram_schmidt_orthonormalization import gram_schmidt_orthonormalization
from sla.random.generate_random_symmetric_positive_definite_matrix import generate_random_symmetric_positive_definite_matrix


def test_simultaneous_iteration_lanczos():
    np.random.seed(40)

    n = 6
    a_matrix = generate_random_symmetric_positive_definite_matrix(n)
    x0 = np.random.randn(n)
    n_eigen_values = 2
    m = 2*n_eigen_values
    eigen_values, eigen_vectors = simultaneous_iteration_lanczos(a_matrix, n_eigen_values, x0=x0, m=m, verbose=True)
    assert len(eigen_values) == m
    assert is_orthogonal_matrix(eigen_vectors)
    np.testing.assert_almost_equal(
        a_matrix @ eigen_vectors,
        eigen_vectors @ np.diag(eigen_values),
        decimal=1
    )