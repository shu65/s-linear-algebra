import numpy as np


def shifted_cocg(A, b, sigma, eps=1e-6):
    x_nm1 = np.zeros(A.shape[1], dtype=A.dtype)
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
    A_p_sigma = A + sigma * np.identity(A.shape[0], dtype=A.dtype)
    x_sigma_nm1 = np.zeros(A.shape[1], dtype=A.dtype)
    p_sigma_nm1 = np.zeros(A.shape[1], dtype=A.dtype)
    r_sigma_nm1 = b
    alpha_sigma_nm1 = 1
    beta_sigma_nm1 = 0
    while (np.linalg.norm(r_nm1) > eps*b_norm) or (np.linalg.norm(r_sigma_nm1) > eps*b_norm):
        print(iter, "r:", np.linalg.norm(r_nm1), r_nm1, flush=True)
        print(iter, "r_sigma:", np.linalg.norm(r_sigma_nm1), r_sigma_nm1, flush=True)
        x_n = x_nm1 + alpha_nm1 * p_nm1
        p_n = r_n + beta_nm1 * p_nm1
        A_p_n =  A @ p_n

        alpha_n = np.vdot(r_n, r_n) / np.vdot(p_n, A_p_n)
        #r_np1 = -alpha_n * (A @ r_n) + (1 + beta_nm1*alpha_n/alpha_nm1) * r_n - beta_nm1*alpha_n/alpha_nm1 * r_nm1
        r_np1 = r_n - alpha_n * (A @ p_n)
        beta_n = np.vdot(r_np1, r_np1) / np.vdot(r_n, r_n)

        x_sigma_n = x_sigma_nm1 + alpha_sigma_nm1*p_sigma_nm1
        actual_r_sigma_n = b - A_p_sigma @ x_sigma_n
        r_sigma_n = 1/pi_sigma_n*r_n
        p_sigma_n = r_sigma_n + beta_sigma_nm1*p_sigma_nm1
        #p_sigma_n = 1/pi_sigma_n*r_n + beta_sigma_nm1*p_sigma_nm1

        pi_sigma_np1 = (1 + alpha_n*sigma + beta_nm1*alpha_n/alpha_nm1)*pi_sigma_n - beta_nm1*alpha_n/alpha_nm1*pi_sigma_nm1

        alpha_sigma_n = pi_sigma_n/pi_sigma_np1*alpha_n
        beta_sigma_n = (pi_sigma_n/pi_sigma_np1)**2 * beta_n

        x_nm1 = x_n
        r_nm1 = r_n
        r_n = r_np1
        p_nm1 = p_n
        alpha_nm1 = alpha_n
        beta_nm1 = beta_n

        x_sigma_nm1 = x_sigma_n
        r_sigma_nm1 = r_sigma_n
        #r_sigma_n = r_sigma_np1
        p_sigma_nm1 = p_sigma_n
        alpha_sigma_nm1 = alpha_sigma_n
        beta_sigma_nm1 = beta_sigma_n
        pi_sigma_nm1 = pi_sigma_n
        pi_sigma_n = pi_sigma_np1

        iter += 1
    return x_nm1, x_sigma_nm1