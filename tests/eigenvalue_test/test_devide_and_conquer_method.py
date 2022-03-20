import numpy as np

from sla.eigenvalue.devide_and_conquer_method import devide, get_characteristic_polynomial, newtons_method, compute_eigen_values_with_characteristic_polynomial, compute_eigen_vectors_with_characteristic_polynomial, devide_and_conquer_method
from sla.utility import is_orthogonal_matrix


def test_devide():
    a_matrix = np.array([
        [1., 1., 0., 0., ],
        [1., 3., 2., 0., ],
        [0., 2., 1., 3., ],
        [0., 0., 3., 4., ],
    ])

    t1, t2, beta = devide(a_matrix)
    np.testing.assert_almost_equal(
        t1,
        np.array([
            [1., 1.],
            [1., 1.],
        ])
    )
    np.testing.assert_almost_equal(
        t2,
        np.array([
            [-1., 3.],
            [3., 4.],
        ])
    )
    assert beta == 2.0


def test_characteristic_polynomial():
    beta = 2.0
    d = np.array([ 0.,          2. ,        -2.40512484,  5.40512484,])
    w = np.array([ 0.70710678,  0.70710678, -0.90558942,  0.4241554, ])
    f, df = get_characteristic_polynomial(d=d, beta=beta, w=w)
    eps = 1e-4
    x0 = 1.0
    expected_df = (f(x0 + eps) - f(x0)) / eps
    actual_df = df(x0)
    np.testing.assert_allclose(actual_df, expected_df, rtol=1e-3, atol=1e-3)


def test_newtons_method():
    beta = 2.0
    d = np.array([0.,          2. ,        -2.40512484,  5.40512484,])
    w = np.array([0.70710678,  0.70710678, -0.90558942,  0.4241554, ])
    f, df = get_characteristic_polynomial(d=d, beta=beta, w=w)
    x0 = 1.0
    x = newtons_method(f=f, df=df, x0=x0)
    np.testing.assert_allclose(x, 0.74078592, rtol=1e-3, atol=1e-3)


def test_compute_eigen_values_with_characteristic_polynomial():
    beta = 2.0
    d = np.array([0.,          2. ,        -2.40512484,  5.40512484,])
    w = np.array([0.70710678,  0.70710678, -0.90558942,  0.4241554, ])
    eigen_values = compute_eigen_values_with_characteristic_polynomial(d=d, beta=beta, w=w)
    np.testing.assert_allclose(eigen_values, [-1.57162996,  0.74078592,  3.5629302,   6.26791384], rtol=1e-3, atol=1e-3)


def test_compute_eigen_vectors_with_characteristic_polynomial():
    d = np.array([0.,          2. ,        -2.40512484,  5.40512484,])
    eigen_values = np.array([-1.57162996,  0.74078592,  3.5629302,   6.26791384])
    w = np.array([0.70710678,  0.70710678, -0.90558942,  0.4241554, ])
    eigen_vectors = compute_eigen_vectors_with_characteristic_polynomial(d=d, eigen_values=eigen_values, w=w)
    np.testing.assert_allclose(
        eigen_vectors,
        np.array([[ 0.376796, -0.831571, -0.350773, -0.2085  ],
                  [ 0.165802,  0.489207, -0.79964 , -0.306206],
                  [ 0.909913,  0.250779,  0.268193,  0.192977],
                  [ 0.050915,  0.079221,  0.406948, -0.908584]]),
        rtol=1e-3,
        atol=1e-3)


def test_devide_and_conquer_method():
    a_matrix = np.array([
        [1., 1., 0., 0., ],
        [1., 3., 2., 0., ],
        [0., 2., 1., 3., ],
        [0., 0., 3., 4., ],
    ])

    eigen_values, eigen_vectors = devide_and_conquer_method(a_matrix)

    assert len(eigen_values) == 4
    np.testing.assert_almost_equal(np.trace(a_matrix), np.sum(eigen_values))
    assert is_orthogonal_matrix(eigen_vectors)
    np.testing.assert_almost_equal(a_matrix, eigen_vectors @ np.diag(eigen_values) @ eigen_vectors.T)