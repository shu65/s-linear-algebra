import numpy as np


def generate_random_symmetric(n):
    a = np.random.uniform(0, 1, (n, n))
    return a + a.T - np.diag(a.diagonal())
