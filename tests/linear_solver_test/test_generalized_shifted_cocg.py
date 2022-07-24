import numpy as np

from sla.linear_solver.generalized_shifted_cocg import generalized_shifted_cocg
from sla.linear_solver.shifted_cocg import shifted_cocg


def test_generalized_shifted_cocg():
    a_matrix = np.array([[6.0, 4.0, 1.0],
                         [4.0, 8.0, 2.0],
                         [1.0, 2.0, 1.0]])

    b_matrix = np.array([[1.0, 2.0, 0.0],
                         [2.0, 1.0, 2.0],
                         [0.0, 2.0, 1.0]])
    """
    b_matrix = np.identity(a_matrix.shape[0], dtype=a_matrix.dtype)
    """
    inv_b_matrix = np.linalg.pinv(b_matrix)
    print("inv b matrix", inv_b_matrix)
    b = np.array([1.0, 0.0, 0.0])
    sigma = 0.1
    """
    expected_x, expected_x_sigma = shifted_cocg(
        A=inv_b_matrix @ a_matrix,
        b=inv_b_matrix @ b,
        sigma=sigma
    )
    #print("inv_b_matrix @ a_matrix", inv_b_matrix @ a_matrix)
    #print("inv_b_matrix @ b", inv_b_matrix @ b)
    #print("expected_x", expected_x)
    #print("expected_x_sigma", expected_x_sigma)
    """
    x, x_sigma = generalized_shifted_cocg(A=a_matrix, B=b_matrix, b=b, sigma=sigma)
    np.testing.assert_allclose((a_matrix + 0 * b_matrix)  @ x, b, atol=1e-7, rtol=1e-7)

    inb_b_a_matrix_p_sigma = inv_b_matrix @ a_matrix + sigma * np.identity(a_matrix.shape[0], dtype=a_matrix.dtype)
    np.testing.assert_allclose((a_matrix + sigma * b_matrix)  @ x_sigma, b, atol=1e-7, rtol=1e-7)