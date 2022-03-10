import numpy as np

from sla.lu_decompose import lu_decompose_v1, lu_decompose_v2


def test_lu_decompose_v1():
    a_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 1.,],
        [0., 1., 1.],
    ])
    l, u = lu_decompose_v1(a_matrix)
    expected_l = np.array(
        [[1. , 0. , 0. ],
         [0. , 1. , 0. ],
         [0. , 0.5, 1. ]]
    )
    np.testing.assert_equal(l, expected_l)

    expected_u = np.array(
        [[1. , 0. , 0. ],
         [0. , 2. , 1. ],
         [0. , 0. , 0.5]]
    )
    np.testing.assert_equal(u, expected_u)