import numpy as np

from sla.eigenvalue.cholesky_lr_method import cholesky_lr_method


def test_lr_method():
    a_matrix = np.array([[6.0, 4.0, 1.0],
                         [4.0, 8.0, 2.0],
                         [1.0, 2.0, 1.0]])
    eigen_values = cholesky_lr_method(a_matrix, verbose=True)
    assert len(eigen_values) == 3
    np.testing.assert_almost_equal(np.trace(a_matrix), np.sum(eigen_values))