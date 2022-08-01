import numpy as np
import scipy.linalg

from sla.eigenvalue.feast import feast
from sla.utility import is_orthogonal_matrix
from sla.random.generate_random_symmetric_positive_definite_matrix import generate_random_symmetric_positive_definite_matrix


def test_feast():
    np.random.seed(40)

    n = 6
    a_matrix = generate_random_symmetric_positive_definite_matrix(n)
    b_matrix = generate_random_symmetric_positive_definite_matrix(n)
    expected_eigen_values, expected_eigen_vectors = scipy.linalg.eigh(a_matrix, b_matrix)
    sorted_w = np.sort(expected_eigen_values)
    print("sorted_w", sorted_w)
    eps = 1e-3
    lmd_min = sorted_w[3] - eps
    lmd_max = sorted_w[4] + eps
    assert lmd_min < lmd_max
    m = 2
    m0 = m + 1
    n_e = 8
    conv = 1e-5
    eigen_values, eigen_vectors = feast(A=a_matrix, B=b_matrix, m0=m0,  lmd_min=lmd_min, lmd_max=lmd_max, n_e=n_e, verbose=True, conv=conv)
    assert len(eigen_values) >= m
    assert is_orthogonal_matrix(eigen_vectors, b=b_matrix)
    np.testing.assert_allclose(
        a_matrix @ eigen_vectors,
        b_matrix @ eigen_vectors @ np.diag(eigen_values),
        atol=1e-4, rtol=1e-4,
    )