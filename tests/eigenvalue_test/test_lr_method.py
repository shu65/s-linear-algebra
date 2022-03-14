import numpy as np

from sla.eigenvalue.lr_method import lr_method


def test_lr_method():
    a_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 1.,],
        [0., 1., 1.],
    ])
    eigen_values = lr_method(a_matrix, verbose=True)
    assert len(eigen_values) == 3
    np.testing.assert_almost_equal(np.trace(a_matrix), np.sum(eigen_values))