import numpy as np

from sla.random.generate_random_symmetric import generate_random_symmetric
from sla.eigenvalue.bisection_method import bisection_method, bisection_method_for_tridiagonal


def test_bisection_method_for_tridiagonal():
    a_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 1.,],
        [0., 1., 1.],
    ])
    eigen_values = bisection_method_for_tridiagonal(a_matrix, eps=1e-5)
    assert len(eigen_values) == 3
    np.testing.assert_almost_equal(eigen_values, [2.618034, 1.0, 0.381966], decimal=5)


def test_bisection_method():
    np.random.seed(40)
    a_matrix = generate_random_symmetric(4)
    expected_eigen_values, _ = np.linalg.eig(a_matrix)
    eigen_values = bisection_method(a_matrix, eps=1e-6)
    assert len(eigen_values) == 4
    np.testing.assert_almost_equal(eigen_values, sorted(expected_eigen_values, reverse=True), decimal=5)
