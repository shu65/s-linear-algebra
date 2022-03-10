import numpy as np

from sla.gram_schmidt_orthonormalization import gram_schmidt_orthonormalization
from sla.utility import is_orthogonal_matrix

def test_qr_decompose_modified_gram_schmidt():
    a_matrix = np.array([
        [1., 0., 0.,],
        [0., 2., 1.,],
        [0., 1., 1.],
    ])
    q = gram_schmidt_orthonormalization(a_matrix)
    assert is_orthogonal_matrix(q)
