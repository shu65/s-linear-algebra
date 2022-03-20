import numpy as np
from sla.eigenvalue.jacobi_method import jacobi_method
from sla.utility import is_orthogonal_matrix


def test_jacobi_method():
    a_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 1.,],
        [0., 1., 1.],
    ])
    eigen_values, eigen_vectors = jacobi_method(a_matrix, eps=1e-7)
    assert len(eigen_values) == 3
    np.testing.assert_almost_equal(np.trace(a_matrix), np.sum(eigen_values))
    assert is_orthogonal_matrix(eigen_vectors)
    np.testing.assert_almost_equal(a_matrix, eigen_vectors @ np.diag(eigen_values) @ eigen_vectors.T)