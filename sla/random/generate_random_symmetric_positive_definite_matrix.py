import numpy as np


def generate_random_symmetric_positive_definite_matrix(n, dtype=np.float64):
    a = np.random.uniform(0, 1, (n, n)).astype(dtype=dtype)
    return a @ a.T
