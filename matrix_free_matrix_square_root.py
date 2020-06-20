import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from scipy.special import *


def make_matrix_square_root_applier(A, check_error=True):
    A = A.tocsr()
    weights, poles, _ = matrix_inverse_square_root_rational_weights_and_poles(A)
    N = len(weights)

    AA = [A - poles[k]*sps.eye(A.shape[0]) for k in range(N)]
    AA_solvers = [spla.factorized(A) for A in AA]

    def apply_isqrtA(v):
        u = np.zeros(v.shape)
        for k in range(N):
            u = u + weights[k] * AA_solvers[k](v)
        return u

    def apply_sqrtA(v):
        return A * apply_isqrtA(v)

    if check_error:
        v = np.random.randn(A.shape[1])
        u1 = apply_sqrtA(apply_sqrtA(v))
        u2 = A * v
        sqrt_err = np.linalg.norm(u1-u2)/np.linalg.norm(u2)
        print('sqrt_err=', sqrt_err)

    return apply_sqrtA


def matrix_inverse_square_root_rational_weights_and_poles(A, tol=1e-8, N_max=30, verbose=False):
    # Computes the weights and poles for a rational approximation to the inverse square root of a
    # symmetric positive definite matrix.
    max_lambda = spla.eigsh(A, 1, return_eigenvectors=False, which='LM')[0]
    min_lambda = spla.eigsh(A, 1, return_eigenvectors=False, which='SM')[0]
    if verbose:
        print('min_lambda=', min_lambda, ', max_lambda=', max_lambda)

    xx = np.logspace(np.log10(min_lambda), np.log10(max_lambda), 1000)
    for N in range(1,N_max+1):
        weights, poles, rational_function = inverse_square_root_rational_weights_and_poles(min_lambda, max_lambda, N)
        yy = rational_function(xx)
        yy_true = 1./np.sqrt(xx)
        err = np.linalg.norm(yy_true - yy)/np.linalg.norm(yy_true)
        if verbose:
            print('N=', N, ', err=', err)
        if err < tol:
            break
    return weights, poles, rational_function


def inverse_square_root_rational_weights_and_poles(lower_limit_m, upper_limit_M, number_of_rational_terms_N):
    # Computes weights and poles for the rational approximation
    #   1/sqrt(z) = weights[0]/(z - poles[0]) + weights[1]/(z - poles[1]) + ... + weights[N]/(z - poles[N])
    # designed to be accurate on the positive interval [m,M].
    #
    # Adapted version of method 3 from:
    #   Hale, Nicholas, Nicholas J. Higham, and Lloyd N. Trefethen.
    #   "Computing A^α,\log(A), and related matrix functions by contour integrals."
    #   SIAM Journal on Numerical Analysis 46.5 (2008): 2505-2523.
    m = lower_limit_m
    M = upper_limit_M
    N = number_of_rational_terms_N
    k2 = m/M
    Kp = ellipk(1-k2)
    t = 1j * np.arange(0.5, N) * Kp/N
    sn, cn, dn, ph = ellipj(t.imag,1-k2)
    cn = 1./cn
    dn = dn * cn
    sn = 1j * sn * cn
    w = np.sqrt(m) * sn
    dzdt = cn * dn

    poles = (w**2).real
    weights = (2 * Kp * np.sqrt(m) / (np.pi*N)) * dzdt
    rational_function = lambda zz: np.dot(1. / (zz.reshape((-1,1)) - poles), weights)
    return weights, poles, rational_function