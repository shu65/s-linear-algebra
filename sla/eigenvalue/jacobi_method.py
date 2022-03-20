import numpy as np

def argmax_non_diag_abs(a):
    abs_a = np.abs(a)
    n = a.shape[0]
    for i in range(n):
        abs_a[i, i] = 0
    idx = np.argmax(abs_a)
    if idx == 0:
        return None, None
    else:
        p = idx // n
        q = idx % n
        return p, q


def givens_transformation(a, p, q):
    n = a.shape[0]
    aa = a.copy()

    cot_2_theta = (a[q,q] - a[p,p]) / (2*a[p, q])
    tan_theta = np.sign(cot_2_theta) / (np.abs(cot_2_theta) + np.sqrt(1 + cot_2_theta**2))
    cos_theta = 1 / np.sqrt(1 + tan_theta ** 2)
    sin_theta = cos_theta * tan_theta
    tan_theta_per_2 = sin_theta / (1 + cos_theta)

    aa[p, p] = a[p, p] - tan_theta * a[p, q]
    aa[q, q] = a[q, q] + tan_theta * a[p, q]

    aa[p, q] = 0
    aa[q, p] = 0
    for j in range(n):
        if (j == p) or (j == q):
            continue
        aa[p, j] = a[p, j] - sin_theta * (a[q, j] + tan_theta_per_2 * a[p, j])
        aa[j, p] = aa[p, j]
        aa[q, j] = a[q, j] + sin_theta * (a[p, j] + tan_theta_per_2 * a[q, j])
        aa[j, q] = aa[q, j]

    g = np.identity(n)
    g[p, p] = cos_theta
    g[p, q] = sin_theta
    g[q, p] = - sin_theta
    g[q, q] = cos_theta
    return aa, g,


def jacobi_method(a, eps=1e-5):
    n = a.shape[0]
    eigen_values = a.copy()
    eigen_vectors = np.identity(n)
    next_p, next_q = argmax_non_diag_abs(eigen_values)
    if next_p is not None:
        next_non_diag_abs_a = np.abs(eigen_values[next_p, next_q])
        while next_non_diag_abs_a > eps:
            eigen_values, g = givens_transformation(eigen_values, next_p, next_q)
            eigen_vectors = np.dot(eigen_vectors, g)
            next_p, next_q = argmax_non_diag_abs(eigen_values)
            if next_p is None:
                break
            next_non_diag_abs_a = np.abs(eigen_values[next_p, next_q])
    return np.diag(eigen_values), eigen_vectors