import numpy as np

from sla.eigenvalue.qr_method import qr_method


def test_qr_method():
    a_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 1.,],
        [0., 1., 1.],
    ])
    eigen_values = qr_method(a_matrix, verbose=True)
    assert len(eigen_values) == 3
    np.testing.assert_almost_equal(np.trace(a_matrix), np.sum(eigen_values))