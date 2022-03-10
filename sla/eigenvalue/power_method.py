import numpy as np


def power_method(a, max_iterations=100, verbose=False):
    eigen_vector = np.ones((a.shape[1],))/np.sqrt(a.shape[1])
    for i in range(max_iterations):
        y = a @ eigen_vector
        eigen_value = np.linalg.norm(y)
        eigen_vector = y/eigen_value
        if verbose:
            print(f"iter:{i} eigen_value:{eigen_value} eigen_vector:{eigen_vector}")
    return eigen_value, eigen_vector