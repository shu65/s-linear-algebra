import numpy as np

from sla.eigenvalue.power_method import power_method

def test_power_method():
    a_matrix = np.array(
        [[1, 0],
         [0, 2]]
    )
    expected_eigen_value, expected_eigen_vector = np.linalg.eig(a_matrix)
    eigen_value, eigen_vector = power_method(a_matrix)
    np.testing.assert_almost_equal(eigen_value, expected_eigen_value)
    np.testing.assert_almost_equal(eigen_vector, expected_eigen_vector)