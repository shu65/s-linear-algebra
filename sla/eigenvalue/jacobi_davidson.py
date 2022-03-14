import numpy as np

from sla.eigenvalue.power_method import power_method

def jacobi_davidson(a, x0, max_k, eps=1e-3, max_iterations=100, verbose=False):
    n = a.shape[1]
    v = np.zeros((n, max_k))
    v0 = x0 / np.linalg.norm(x0)
    v[:, 0] = v0
    identity_n = np.identity(n)
    for k in range(max_k):
        v_k = v[:, 0:k + 1]
        b = v_k.T @ a @ v_k
        theta_k, y_k = power_method(b, max_iterations=max_iterations, verbose=verbose)
        u_k = v_k @ y_k
        r_k = a @ u_k - theta_k * u_k
        if np.linalg.norm(r_k) < eps:
            break
        uu = u_k @ u_k.T
        c = (identity_n - uu) @ (a - theta_k*identity_n) @ (identity_n - uu)
        t_k = np.linalg.solve(c, -r_k)
        np.testing.assert_almost_equal(t_k.T @ r_k, 0)
    eigen_value = theta_k
    eigen_vector = u_k
    return eigen_value, eigen_vector