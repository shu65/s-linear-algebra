import numpy as np

from sla.linear_solver.cg import cg


def test_cg():
    a_matrix = np.array([[6.0, 4.0, 1.0],
                         [4.0, 8.0, 2.0],
                         [1.0, 2.0, 1.0]])
    b = np.array([1.0, 0.0, 0.0])
    x0 = np.ones(a_matrix.shape[1], dtype=a_matrix.dtype)
    x = cg(A=a_matrix, b=b)
    np.testing.assert_almost_equal(a_matrix @ x, b)