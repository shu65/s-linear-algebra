import numpy as np

from sla.ldl_t_decompose import ldl_t_decompose, ldl_t_decompose_for_tridiagonal_matrix


def test_ldl_t_decompose():
    expected_l_matrix = np.array([
        [1., 0., 0.,],
        [2., 1., 0.,],
        [3., 2., 1.],
    ])
    expected_d_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 0.,],
        [0., 0., 3.],
    ])
    a_matrix = expected_l_matrix @ expected_d_matrix @ expected_l_matrix.T
    l, d = ldl_t_decompose(a_matrix)
    np.testing.assert_almost_equal(l @ d @ l.T, a_matrix)
    np.testing.assert_almost_equal(l, expected_l_matrix)
    np.testing.assert_almost_equal(d, expected_d_matrix)


def test_ldl_t_decompose_for_tridiagonal_matrix():
    a_matrix = np.array([
        [1., 2., 0.,],
        [2., 3., 2.,],
        [0., 2., 1.],
    ])

    l, d = ldl_t_decompose_for_tridiagonal_matrix(a_matrix)

    np.testing.assert_almost_equal(l @ d @ l.T, a_matrix)
