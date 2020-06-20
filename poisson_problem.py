import numpy as np
from fenics import *
import scipy.sparse.linalg as spla
from scipy.special import *
from matrix_free_matrix_square_root import make_matrix_square_root_applier
from fenics_helper_functions import fenics_to_scipy_matrix_conversion, random_function
from higher_derivatives_action import HigherDerivativesAction


def nonlinear_neumann_poisson_problem(n):
    # State equation:
    #  div(exp(m)grad(u)) + u^3 = f in [0,1]^2
    #  normal . exp(m)grad(u) = 0   on boundary of [0,1]^2
    mesh = UnitSquareMesh(n,n)
    M = FunctionSpace(mesh, 'CG', 1)
    V = M

    m = Function(M)
    u = Function(V)
    v = TestFunction(V)

    sigma_f = 0.2
    xx = M.tabulate_dof_coordinates()
    f = Function(M)
    f.vector()[:] = np.exp(np.sum(-0.5 * ((xx - 0.5) / sigma_f) ** 2, axis=1))

    Q = u * TestFunction(M) * ds  # quantity of interest

    a = (exp(m) * inner(grad(u), grad(v)) + v * u**3) * dx - f * v * dx # PDE

    bcs = []

    u.vector()[:] = 1.0

    QHD = HigherDerivativesAction(Q, a, bcs, m, u)
    apply_sqrtC, apply_sqrtC_T = make_prior_preconditioner_and_adjoint(M) # square root of covariance operator
    apply_preconditioned_QHD, _, _ = make_prior_preconditioned_derivative_tensor_appliers(QHD, apply_sqrtC, apply_sqrtC_T)

    def forward_map(new_m_vec):
        old_m_vec = m.vector()[:].copy()
        m.vector()[:] = new_m_vec

        q = apply_preconditioned_QHD([], 0)

        m.vector()[:] = old_m_vec
        return q


    return Q, a, m, u, bcs, apply_sqrtC, apply_sqrtC_T, QHD, apply_preconditioned_QHD, forward_map


def make_prior_preconditioner_and_adjoint(M, check_adjoint_correctness=True):
    # Square root of covariance operator, C, and transpose
    # C = (Laplacian + I)^(-2)
    vt = TestFunction(M)
    ut = TrialFunction(M)

    C_form = inner(grad(vt), grad(ut)) * dx + vt * ut * dx
    C = fenics_to_scipy_matrix_conversion(assemble(C_form))
    solve_C = spla.factorized(C)

    W_form = ut*vt*dx
    W = fenics_to_scipy_matrix_conversion(assemble(W_form))
    apply_sqrtW = make_matrix_square_root_applier(W)

    def apply_sqrtC(f_vec):
        return solve_C(apply_sqrtW(f_vec))

    def apply_sqrtC_T(u_vec):
        return apply_sqrtW(solve_C(u_vec))

    if check_adjoint_correctness:
        xl = np.random.randn(M.dim())
        xr = np.random.randn(M.dim())
        smooth_vector_adjoint_err = np.dot(apply_sqrtC(xl).copy(), xr) - np.dot(xl, apply_sqrtC_T(xr).copy())
        print('prior_adjoint_err=', smooth_vector_adjoint_err)

    return apply_sqrtC, apply_sqrtC_T


def make_prior_preconditioned_derivative_tensor_appliers(QHD, apply_sqrtC, apply_sqrtC_T, check_consistency=True):
    M = QHD.M
    Z = QHD.Z
    pp = [Function(M) for _ in range(10)]
    def apply_T_derivative_mode_free(pp_vecs, z_vec):
        k = len(pp_vecs)
        for p, p_vec in zip(pp[:k], pp_vecs[:k]):
            p.vector()[:] = apply_sqrtC(p_vec)
        z = Function(Z)
        z.vector()[:] = z_vec
        gx = QHD.compute_derivative_of_quantity_of_interest(pp[:k], z)
        return apply_sqrtC_T(gx[:])

    def apply_T_output_mode_free(pp_vecs):
        k = len(pp_vecs)
        for p, p_vec in zip(pp[:k], pp_vecs[:k]):
            p.vector()[:] = apply_sqrtC(p_vec)
        qx = QHD.compute_derivative_of_quantity_of_interest(pp[:k], None)
        return qx[:]

    def apply_T(vv, k):
        output_mode_free = (k == len(vv))
        if output_mode_free:
            c = apply_T_output_mode_free(vv)
        else:
            c = apply_T_derivative_mode_free(vv[:-1], vv[-1])
        return c

    if check_consistency:
        _check_firstfree_vs_lastfree_consistency(apply_T_derivative_mode_free, apply_T_output_mode_free, M, Z)

    return apply_T, apply_T_derivative_mode_free, apply_T_output_mode_free


def _check_firstfree_vs_lastfree_consistency(apply_T_derivative_mode_free, apply_T_output_mode_free, M, Z):
    def apply_T_saturated1(pp_vecs, z_vec):
        q = apply_T_derivative_mode_free(pp_vecs[:-1], z_vec)
        return np.dot(q, pp_vecs[-1])

    def apply_T_saturated2(pp_vecs, z_vec):
        x = apply_T_output_mode_free(pp_vecs)
        return np.dot(x, z_vec)

    p1_vec, p1 = random_function(M)
    p2_vec, p2 = random_function(M)
    p3_vec, p3 = random_function(M)
    p4_vec, p4 = random_function(M)

    z_vec, z = random_function(Z)

    k1 = apply_T_saturated1([p1_vec, p2_vec, p3_vec, p4_vec], z_vec)
    k2 = apply_T_saturated2([p1_vec, p2_vec, p3_vec, p4_vec], z_vec)

    first_vs_last_modes_err = np.abs(k2 - k1) / np.abs(k2)
    print('first_vs_last_modes_err=', first_vs_last_modes_err)

