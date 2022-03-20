import numpy as np


def eigen_value_range(a):
    # using Gershgorin circle theorem
    n = a.shape[0]
    r = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                r[i] += np.abs(a[i, j])
    min_value = np.min(np.diag(a) - r)
    max_value = np.max(np.diag(a) + r)
    return min_value, max_value


def eigen_value_range_for_tridiagonal(a):
    # using Gershgorin circle theorem
    n = a.shape[0]
    r = np.zeros(n)
    r[0] += np.abs(a[0, 1])
    for i in range(1, n-1):
        r[i] += np.abs(a[i, i - 1]) + np.abs(a[i, i + 1])
    r[n-1] += np.abs(a[n-1, n-2])
    min_value = np.min(np.diag(a) - r)
    max_value = np.max(np.diag(a) + r)
    return min_value, max_value