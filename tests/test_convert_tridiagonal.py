import numpy as np

from sla.random.generate_random_symmetric import generate_random_symmetric
from sla.convert_tridiagonal import convert_tridiagonal
from sla.utility import is_tridiagonal_matrix


def test_convert_tridiagonal_with_symmetric():
    np.random.seed(40)
    n = 5
    a_matrix = generate_random_symmetric(n)
    q, t = convert_tridiagonal(a_matrix)
    assert is_tridiagonal_matrix(t)
