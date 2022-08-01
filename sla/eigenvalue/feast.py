import numpy as np
import scipy.linalg

from sla.utility import is_orthogonal_matrix

def feast(A, B, m0,  lmd_min, lmd_max, n_e, max_iter=10, conv=1e-3, verbose=False):
    n = A.shape[0]
    Y = np.random.randn(n, m0).astype(dtype=A.dtype)
    Q = np.zeros((n, m0), dtype=A.dtype)
    r = (lmd_max - lmd_min) / 2
    x, w = np.polynomial.legendre.leggauss(n_e)
    print("x", x)
    print("w", w)
    last_lmd_m = np.zeros((0), dtype=A.dtype)
    for i in range(max_iter):
        for e in range(n_e):
            x_e = x[e]
            w_e = w[e]
            theta_e = -(np.pi/2) * (x_e - 1)
            z_e = (lmd_max + lmd_min)/2 + r * np.exp(1j*theta_e)
            Q_e = np.linalg.solve(z_e * B - A, Y)
            Q = Q - (w_e/2)*np.real(r * np.exp(1j*theta_e) * Q_e)

        #assert is_orthogonal_matrix(Q)
        A_Q = Q.T @ A @ Q
        B_Q = Q.T @ B @ Q
        eps, phi = scipy.linalg.eigh(A_Q, B_Q)
        selected_idx = (lmd_min <= eps) & (eps <= lmd_max)
        lmd = eps
        X = Q @ phi

        lmd_m = lmd[selected_idx]
        X_m = X[:, selected_idx]
        # check conv
        if lmd_m.shape == last_lmd_m.shape:
            diff_lmd = np.abs(lmd_m - last_lmd_m)
            if np.all(diff_lmd < conv):
                break

        last_lmd_m = lmd_m
        Y = B @ X
    return lmd_m, X_m