import numpy as np

from sla.linear_solver.cg import cg

def generalized_shifted_cocg(A, B, b, sigma, eps=1e-5):
    x_n = np.zeros(A.shape[1], dtype=A.dtype)
    p_nm1 = np.zeros(A.shape[1], dtype=A.dtype)
    r_nm1 = b
    alpha_nm1 = 1
    beta_nm1 = 0
    b_norm = np.linalg.norm(b)
    iter = 0
    A_p_nm1 = A @ p_nm1
    r_n = r_nm1 - alpha_nm1 * A_p_nm1

    pi_sigma_nm1 = 1
    pi_sigma_n = 1
    x_sigma_n = np.zeros(A.shape[1], dtype=A.dtype)
    p_sigma_nm1 = np.zeros(A.shape[1], dtype=A.dtype)
    r_sigma_n = r_n
    inv_B = np.linalg.pinv(B)
    while (np.linalg.norm(r_n) > eps*b_norm) or (np.linalg.norm(r_sigma_n) > eps*b_norm):
        print(iter, "r:", np.linalg.norm(r_n), r_n, flush=True)
        print(iter, "r_sigma:", np.linalg.norm(r_sigma_n), r_sigma_n, flush=True)
        inv_B_r_n = inv_B @ r_n
        p_n = inv_B_r_n + beta_nm1 * p_nm1
        A_p_n = A @ p_n
        alpha_n = np.vdot(inv_B_r_n, r_n) / np.vdot(p_n, A_p_n)
        x_np1 = x_n + alpha_n * p_n
        r_np1 = r_n - alpha_n * (A @ p_n)
        expected_r_np1 = b - A @ x_np1
        inv_B_r_np1 = inv_B @ r_np1
        beta_n = np.vdot(inv_B_r_np1, r_np1) / np.vdot(inv_B_r_n, r_n)

        pi_sigma_np1 = (1 + alpha_n*sigma + beta_nm1*alpha_n/alpha_nm1)*pi_sigma_n - beta_nm1*alpha_n/alpha_nm1*pi_sigma_nm1
        beta_sigma_nm1 = (pi_sigma_nm1/pi_sigma_n)**2 * beta_nm1
        alpha_sigma_n = pi_sigma_n/pi_sigma_np1*alpha_n
        p_sigma_n = 1/pi_sigma_n*inv_B_r_n + beta_sigma_nm1*p_sigma_nm1
        x_sigma_np1 = x_sigma_n + alpha_sigma_n*p_sigma_n
        r_sigma_np1 = 1/pi_sigma_np1*r_np1
        expected_r_sigma_np1 = b - (A + sigma * B) @ x_sigma_np1
        x_n = x_np1
        r_n = r_np1
        p_nm1 = p_n
        alpha_nm1 = alpha_n
        beta_nm1 = beta_n

        x_sigma_n = x_sigma_np1
        r_sigma_n = r_sigma_np1
        p_sigma_nm1 = p_sigma_n
        pi_sigma_nm1 = pi_sigma_n
        pi_sigma_n = pi_sigma_np1

        iter += 1
    print(iter, "r:", np.linalg.norm(r_n), r_n, flush=True)
    print(iter, "r_sigma:", np.linalg.norm(r_sigma_n), r_sigma_n, flush=True)
    return x_n, x_sigma_n