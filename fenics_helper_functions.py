import numpy as np
from fenics import *
import scipy.sparse as sps


def fenics_to_scipy_matrix_conversion(A_fenics):
    ai, aj, av = as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy


def random_function(X, smooth=False):
    q_fct = Function(X)
    q_fct.vector()[:] = np.random.randn(X.dim())
    if smooth:
        L_form = inner(grad(TestFunction(X)), grad(q_fct)) * dx + TestFunction(X) * q_fct * dx
        solve(L_form == 0, q_fct)
        q0 = np.copy(q_fct.vector()[:])
        q_fct.vector()[:] = (q0 - np.mean(q0)) / (2 * np.std(q0))
    return q_fct.vector()[:], q_fct