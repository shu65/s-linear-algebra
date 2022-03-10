import numpy as np

from sla.qr_decompose import qr_decompose_gram_schmidt, qr_decompose_modified_gram_schmidt
from sla.utility import is_upper_triangular_matrix, is_orthogonal_matrix

def test_qr_decompose_gram_schmidt():
    a_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 1.,],
        [0., 1., 1.],
    ])
    q, r = qr_decompose_gram_schmidt(a_matrix)

    print()
    np.testing.assert_almost_equal(a_matrix, q @ r)
    assert is_orthogonal_matrix(q)
    assert is_upper_triangular_matrix(r)
    np.testing.assert_almost_equal(q.T @ a_matrix, r)


def test_qr_decompose_modified_gram_schmidt():
    a_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 1.,],
        [0., 1., 1.],
    ])
    q, r = qr_decompose_modified_gram_schmidt(a_matrix)
    np.testing.assert_almost_equal(a_matrix, q @ r)
    assert is_orthogonal_matrix(q)
    assert is_upper_triangular_matrix(r)
    np.testing.assert_almost_equal(q.T @ a_matrix, r)