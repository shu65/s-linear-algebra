import numpy as np

from sla.cholesky_decompose import cholesky_decompose


def test_cholesky_decompose():
    a_matrix = np.array([[6.0, 4.0, 1.0],
                         [4.0, 8.0, 2.0],
                         [1.0, 2.0, 1.0]])
    c = cholesky_decompose(a_matrix)
    expected_c = np.linalg.cholesky(a_matrix)
    np.testing.assert_almost_equal(c, expected_c)
    np.testing.assert_almost_equal(c @ c.T, a_matrix)
