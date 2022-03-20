import numpy as np


def devide(a):
    m = a.shape[0] // 2
    beta = a[m, m - 1]
    t1 = a[:m, :m].copy()
    t2 = a[m:, m:].copy()

    t1[m-1, m-1] -= beta
    t2[0, 0] -= beta
    return t1, t2, beta


def get_characteristic_polynomial(d, beta, w):
    def f(lmd):
        tmp = w**2 / (d - lmd)
        ret = 1 + beta * np.sum(tmp)
        return ret

    def df(lmd):
        tmp = w**2 / ((d - lmd)**2)
        ret = beta * np.sum(tmp)
        return ret
    return f, df


def newtons_method(f, df, x0, eps=1e-5, max_iterations=100):
    iterations = 0
    x_n = x0
    x_np1 = x0
    while iterations < max_iterations:
        f_value = f(x_n)
        df_value = df(x_n)
        x_np1 = x_n - f_value / df_value
        if np.abs(x_np1 - x_n) < eps:
            break
        x_n = x_np1
        iterations += 1
    return x_np1


def compute_eigen_values_with_characteristic_polynomial(d, beta, w):
    f, df = get_characteristic_polynomial(d=d, beta=beta, w=w)
    sorted_d = np.sort(d)
    if beta == 0:
        return d
    elif beta > 0.0:
        x0s = (sorted_d[:-1] + sorted_d[1:]) / 2.0
        x0s = np.hstack((x0s, np.array([sorted_d[-1] + 1.0])))
    else:
        x0s = (sorted_d[:-1] + sorted_d[1:]) / 2.0
        x0s = np.hstack((np.array([sorted_d[0] - 1.0], x0s)))
    eigen_values = []
    for x0 in x0s:
        x = newtons_method(f=f, df=df, x0=x0)
        eigen_values.append(x)
    return np.array(eigen_values)


def compute_eigen_vectors_with_characteristic_polynomial(d, eigen_values, w):
    eigen_vectors = []
    for eigen_value in eigen_values:
        eigen_vector = (1/(d - eigen_value)) * w
        eigen_vectors.append(eigen_vector / np.linalg.norm(eigen_vector))
    return np.array(eigen_vectors).T


def devide_and_conquer_method(a):
    n = a.shape[0]
    t1, t2, beta = devide(a)
    d1, q1 = np.linalg.eigh(t1)
    d2, q2 = np.linalg.eigh(t2)
    np.testing.assert_almost_equal(q1 @ np.diag(d1) @ q1.T, t1)
    w = np.hstack((q1[:, -1], q2[:, 0]))
    #assert (w.reshape(n, 1) @ w.reshape(n, 1).T).shape == (n, n)
    d = np.hstack((d1, d2))
    q = np.zeros((n, n))
    q[:q1.shape[0], :q1.shape[1]] = q1
    q[q1.shape[0]:, q1.shape[1]:] = q2
    #b = np.diag(d) + beta * w.reshape(n, 1) @ w.reshape(n, 1).T
    #eigen_values, eigen_vectors = np.linalg.eigh(b)
    eigen_values = compute_eigen_values_with_characteristic_polynomial(d=d, beta=beta, w=w)
    eigen_vectors = compute_eigen_vectors_with_characteristic_polynomial(d=d, eigen_values=eigen_values, w=w)
    return eigen_values, q @ eigen_vectors