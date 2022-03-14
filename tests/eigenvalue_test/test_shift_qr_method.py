import numpy as np

from sla.eigenvalue.shift_qr_method import shift_qr_method


def test_shift_qr_algorithm():
    a_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 1.,],
        [0., 1., 1.],
    ])
    eigen_values = shift_qr_method(a_matrix, s=2.60, verbose=True)
    assert len(eigen_values) == 3
    np.testing.assert_almost_equal(np.trace(a_matrix), np.sum(eigen_values))