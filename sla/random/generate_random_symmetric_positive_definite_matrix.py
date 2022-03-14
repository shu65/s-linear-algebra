import numpy as np


def generate_random_symmetric_positive_definite_matrix(n):
    a = np.random.uniform(0, 1, (n, n))
    return a @ a.T
