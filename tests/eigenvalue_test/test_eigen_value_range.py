import numpy as np

from sla.eigenvalue.eigen_value_range import eigen_value_range, eigen_value_range_for_tridiagonal


def test_eigen_value_range():
    a_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 1.,],
        [0., 1., 1.],
    ])
    min_eigen_value, max_eigen_value = eigen_value_range(a_matrix)
    eigen_values, _ = np.linalg.eig(a_matrix)
    np.testing.assert_array_less(min_eigen_value, np.min(eigen_values))
    np.testing.assert_array_less(np.max(eigen_values), max_eigen_value)


def test_eigen_value_range_for_tridiagonal():
    a_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 1.,],
        [0., 1., 1.],
    ])
    min_eigen_value, max_eigen_value = eigen_value_range_for_tridiagonal(a_matrix)
    eigen_values, _ = np.linalg.eig(a_matrix)
    np.testing.assert_array_less(min_eigen_value, np.min(eigen_values))
    np.testing.assert_array_less(np.max(eigen_values), max_eigen_value)