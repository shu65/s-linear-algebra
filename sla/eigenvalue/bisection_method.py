import numpy as np
from sla.convert_tridiagonal import convert_tridiagonal
from sla.ldlt_decompose import ldlt_decompose_for_tridiagonal_matrix
from sla.eigenvalue.eigen_value_range import eigen_value_range_for_tridiagonal


def bisection_method(a, eps=1e-1):
    _, t = convert_tridiagonal(a)
    expected_eigen_values, _ = np.linalg.eig(t)
    #print("t", t)
    #print("expected_eigen_values t", expected_eigen_values)
    eigen_values = bisection_method_for_tridiagonal(t, eps=eps)
    return eigen_values


def compute_n_eigen_values_greater_than_threshold(t, threshold):
    # using Sylvester
    n = t.shape[0]
    new_t = t - threshold * np.identity(n)
    _, d = ldlt_decompose_for_tridiagonal_matrix(new_t)
    #print("d", d)
    positive_eigen_value_count = np.sum((np.diag(d) > 0)).astype(np.int)
    return positive_eigen_value_count


def bisection_method_for_tridiagonal(t, eps=1e-2):
    min_eigen_value, max_eigen_value = eigen_value_range_for_tridiagonal(t)
    n_eigen_values = compute_n_eigen_values_greater_than_threshold(t, min_eigen_value)
    eigen_values = bisection_method_recursive(
        t=t,
        eps=eps,
        min_eigen_value_in_target=min_eigen_value,
        max_eigen_value_in_target=max_eigen_value,
        min_eigen_value_order=n_eigen_values,
        max_n_eigen_value_order=0,
        n_eigen_values=n_eigen_values,
    )
    return eigen_values


def bisection_method_recursive(t, eps, min_eigen_value_in_target, max_eigen_value_in_target, min_eigen_value_order, max_n_eigen_value_order, n_eigen_values):
    s = (min_eigen_value_in_target + max_eigen_value_in_target) / 2.0
    s_eigen_value_order = compute_n_eigen_values_greater_than_threshold(t, s)
    assert min_eigen_value_order >= max_n_eigen_value_order
    lower_count = min_eigen_value_order - s_eigen_value_order
    if lower_count == 0:
        lower_eigen_values = []
    elif (s - min_eigen_value_in_target) < eps:
        lower_eigen_values = [s for _ in range(lower_count)]
    else:
        lower_eigen_values = bisection_method_recursive(
            t=t,
            eps=eps,
            min_eigen_value_in_target=min_eigen_value_in_target,
            max_eigen_value_in_target=s,
            min_eigen_value_order=min_eigen_value_order,
            max_n_eigen_value_order=s_eigen_value_order,
            n_eigen_values=n_eigen_values,
        )
    upper_count = s_eigen_value_order - max_n_eigen_value_order
    if upper_count == 0:
        upper_eigen_values = []
    elif (max_eigen_value_in_target - s) < eps:
        upper_eigen_values = [s for _ in range(upper_count)]
    else:
        upper_eigen_values = bisection_method_recursive(
            t=t,
            eps=eps,
            min_eigen_value_in_target=s,
            max_eigen_value_in_target=max_eigen_value_in_target,
            min_eigen_value_order=s_eigen_value_order,
            max_n_eigen_value_order=max_n_eigen_value_order,
            n_eigen_values=n_eigen_values,
        )
    return upper_eigen_values + lower_eigen_values
