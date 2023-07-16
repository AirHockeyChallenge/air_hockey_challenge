import numpy as np
from scipy.linalg import qr, svd


def qr_null(A, tol=None):
    Q, R = qr(A.T, mode='full')
    tol = np.max(A) * np.finfo(R.dtype).eps if tol is None else tol
    rnk = min(A.shape) - np.abs(np.diag(R))[::-1].searchsorted(tol)
    return Q[:, rnk:].conj()


def smooth_basis(A, T0=None):
    """
    Compute the null space matrix suggested by:
    On the computation of multidimensional solution manifolds of parametrized equations
    """
    Ux = qr_null(A)
    if T0 is None:
        T0 = np.zeros(Ux.shape)
        np.fill_diagonal(T0, 1.0)
    else:
        assert T0.shape == (Ux.shape[0], Ux.shape[0] - Ux.shape[1])

    U0 = Ux.T @ T0
    U, s, Vh = svd(U0)
    Q = U @ Vh
    return Ux @ Q
