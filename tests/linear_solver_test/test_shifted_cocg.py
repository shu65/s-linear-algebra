import numpy as np

from sla.linear_solver.shifted_cocg import shifted_cocg


def test_shifted_cocg():
    a_matrix = np.array([[6.0, 4.0, 1.0],
                         [4.0, 8.0, 2.0],
                         [1.0, 2.0, 1.0]])
    b = np.array([1.0, 0.0, 0.0])
    sigma = 0.1
    x, x_sigma = shifted_cocg(A=a_matrix, b=b, sigma=0.1)
    np.testing.assert_almost_equal(a_matrix @ x, b)

    a_matrix_p_sigma = a_matrix + sigma * np.identity(a_matrix.shape[0], dtype=a_matrix.dtype)
    np.testing.assert_almost_equal(a_matrix_p_sigma @ x_sigma, b)