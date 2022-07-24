import numpy as np


def cg(A, b, eps=1e-3):
    x0 = np.zeros(A.shape[1], dtype=A.dtype)
    r = b - A @ x0
    p = r
    b_norm = np.linalg.norm(b)
    iter = 0
    x = x0
    while np.linalg.norm(r) > eps*b_norm:
        print("r:", np.linalg.norm(r), r)
        A_p = A @ p
        alpha = np.vdot(r, p) / np.vdot(p, A_p)
        x_next = x + alpha * p
        r_next = r - alpha * A_p
        beta = - np.vdot(r_next, A_p) / np.vdot(p, A_p)
        p_next = r_next + beta * p

        x = x_next
        r = r_next
        p = p_next
        iter += 1
    return x