import numpy as np
from helper_functions import unit_vector
from tensor_operations import third_order_tensor_double_contraction, partial_tensor_train_pushthrough

# See EXAMPLE_hilbert.py for example usage

#### Shared parameters ####

_tensor_train_construction_parameters = dict()


#### Main tensor train approximation function ####

def tensor_train_from_tensor_action(tensor_action, tensor_shape,
                                    step=1, min_rank=2, max_rank=20, tol=1e-6, verbose=True,
                                    oversampling_parameter=5, basis_oversampling_parameter=1):
    # returns CC, the cores of a tensor train approximation of a tensor T
    #
    # T is a N1 x N2 x ... x Nd tensor/array
    # tensor_shape = (N1, N2, ..., Nd)
    #
    # tensor_actions is a function that applies the action of T with the kth mode free.
    #   e.g., for a 4th order tensor T:
    #     tensor_action([y,z,w], 0) = T(., y, z, w)
    #     tensor_action([x,z,w], 1) = T(x, ., z, w)
    #     tensor_action([x,y,w], 2) = T(x, y, ., w)
    #     tensor_action([x,y,z], 3) = T(x, y, z, .)
    #   where
    #     T(x,y,z,w) := \sum_{ijkl} T_{ijkl} x_i y_j z_k w_l
    _tensor_train_construction_parameters['oversampling_parameter'] = oversampling_parameter
    _tensor_train_construction_parameters['basis_oversampling_parameter'] = basis_oversampling_parameter
    _tensor_train_construction_parameters['step'] = step
    _tensor_train_construction_parameters['tol'] = tol
    _tensor_train_construction_parameters['min_rank'] = min_rank
    _tensor_train_construction_parameters['max_rank'] = max_rank
    _tensor_train_construction_parameters['verbose'] = verbose

    d = len(tensor_shape)

    CC = []
    CC.append(_compute_first_core(tensor_action, tensor_shape))
    CC.append(_compute_second_core(tensor_action, tensor_shape, CC))
    while len(CC) < d-1:
        CC.append(_compute_intermediate_core(tensor_action, tensor_shape, CC))
    CC.append(_compute_last_core(tensor_action, tensor_shape, CC))

    return CC


#### Functions for computing cores ####

def _compute_first_core(tensor_action, tensor_shape):
    core_index = 0
    if _tensor_train_construction_parameters['verbose']:
        print('Core index=', core_index)

    def F(ww):
        return tensor_action(ww, core_index)

    C1 = _construct_core_with_randomized_range_finder(F, tensor_shape, core_index, 1)
    return C1


def _compute_second_core(tensor_action, tensor_shape, CC):
    core_index = 1
    if _tensor_train_construction_parameters['verbose']:
        print('Core index=', core_index)

    C1 = CC[-1]
    r1 = C1.shape[-1]
    N2 = tensor_shape[core_index]

    def F(ww):
        y = np.zeros((r1, N2))
        for j in range(r1):
            eta_j = C1[0, :, j]
            T_args = [eta_j] + ww
            y[j,:] = tensor_action(T_args, core_index)
        return y.reshape(-1)

    C2 = _construct_core_with_randomized_range_finder(F, tensor_shape, core_index, r1)
    return C2


def _compute_intermediate_core(tensor_action, tensor_shape, CC):
    core_index = len(CC)
    if _tensor_train_construction_parameters['verbose']:
        print('Core index=', core_index)

    all_psi, all_xi, all_eta = _compute_special_input_vectors(CC)

    _, _, r_left_next = CC[-1].shape
    N_next = tensor_shape[core_index]
    def F(ww):
        y = np.zeros((r_left_next, N_next))
        for i in range(len(all_xi)):
            for j in range(r_left_next):
                T_args = all_psi + [all_xi[i], all_eta[i][j]] + ww
                y[j,:] += tensor_action(T_args, core_index)
        return y.reshape(-1)

    C_next = _construct_core_with_randomized_range_finder(F, tensor_shape, core_index, r_left_next)
    return C_next


def _compute_last_core(tensor_action, tensor_shape, CC):
    core_index = len(CC)
    if _tensor_train_construction_parameters['verbose']:
        print('Core index=', core_index)

    all_psi, all_xi, all_eta = _compute_special_input_vectors(CC)
    r = CC[-1].shape[2]
    N = tensor_shape[-1]
    C_last = np.zeros((r,N,1))
    for i in range(len(all_xi)):
        for j in range(r):
            T_args = all_psi + [all_xi[i], all_eta[i][j]]
            C_last[j, :, 0] += tensor_action(T_args, core_index)
    return C_last


#### Helper functions ####

def _multilinear_randomized_range_finder_parameterized(F, input_shape):
    return multilinear_randomized_range_finder(F, input_shape,
                                               min_rank=_tensor_train_construction_parameters['min_rank'],
                                               max_rank=_tensor_train_construction_parameters['max_rank'],
                                               oversampling_parameter=_tensor_train_construction_parameters['oversampling_parameter'],
                                               step=_tensor_train_construction_parameters['step'],
                                               tol=_tensor_train_construction_parameters['tol'],
                                               verbose=_tensor_train_construction_parameters['verbose'])


def multilinear_randomized_range_finder(F, input_shape, min_rank=2, max_rank=20, oversampling_parameter=5, step=1, tol=1e-9, verbose=True):
    # Constructs orthonormal basis for the range of vector valued multilinear function
    #   F : V1 x V2 x ... x Vm -> V0
    # where
    #   input_shape = (dim(V1), dim(V2), ..., dim(Vm))
    yy = []
    for _ in range(oversampling_parameter + min_rank):
        ww = list([np.random.randn(N) for N in input_shape])
        yy.append(F(ww))

    for r in range(min_rank, max_rank + step, step):
        Y = np.array(yy).T
        Q0, _, _ = np.linalg.svd(Y, full_matrices=False)
        Q = Q0[:, :r]

        norm_estimate = np.linalg.norm(Y)/np.sqrt(len(yy))
        error_estimate = np.max(np.linalg.norm(Y - np.dot(Q, np.dot(Q.T, Y)), axis=0))/norm_estimate
        if verbose:
            print('r=', r, 'error_estimate=', error_estimate)
        if error_estimate < tol:
            break

        for _ in range(step):
            ww = list([np.random.randn(N) for N in input_shape])
            yy.append(F(ww))

    return Q


def _construct_core_with_randomized_range_finder(F, tensor_shape, core_index, input_rank):
    F_input_shape = tensor_shape[core_index + 1:]
    Ur = _multilinear_randomized_range_finder_parameterized(F, F_input_shape)
    C = Ur.reshape((input_rank, tensor_shape[core_index], -1))
    return C


def _compute_tau(C):
    r_left, N, r_right = C.shape
    return int(np.ceil(float(r_right) / float(N))) + _tensor_train_construction_parameters['basis_oversampling_parameter']


def _compute_special_input_vectors(CC):
    all_psi = [C[0,:,0] for C in CC[:-2]]
    psi_pushthrough = partial_tensor_train_pushthrough(CC[:-2], all_psi).reshape(-1)

    tau = _compute_tau(CC[-2])
    all_xi = [CC[-2][0,:,i] for i in range(tau)]
    psi_xi_pushthroughs = [third_order_tensor_double_contraction(CC[-2], psi_pushthrough, xi, None) for xi in all_xi]

    r_left, N, r_right = CC[-1].shape
    AA = [np.dot(z, CC[-1].reshape((r_left,-1))).reshape((N, r_right)) for z in psi_xi_pushthroughs]
    A = np.bmat([[A.T for A in AA]])

    B = np.eye(r_right)

    ETA = np.linalg.lstsq(A, B, rcond=None)[0]

    least_squares_error = np.linalg.norm(np.dot(A, ETA) - B)
    if least_squares_error > 1e-10:
        print('bad least squares in compute_intermediate_core(). least_squares_error=', least_squares_error)

    ETA = ETA.reshape((tau, N, r_right))
    all_eta = [[ETA[i, :, j].reshape(-1) for j in range(r_right)] for i in range(tau)]

    _check_special_input_vectors(CC, all_psi, all_xi, all_eta)

    return all_psi, all_xi, all_eta


def _check_special_input_vectors(CC, all_psi, all_xi, all_eta):
    tau = len(all_xi)
    r = CC[-1].shape[-1]
    errors = np.zeros(r)
    for j in range(r):
        ej_true = unit_vector(j, r)
        ej = np.zeros(r)
        for i in range(tau):
            xx = all_psi + [all_xi[i], all_eta[i][j]]
            ej += partial_tensor_train_pushthrough(CC, xx).reshape(-1)
        errors[j] = np.linalg.norm(ej - ej_true)
    total_error = np.linalg.norm(errors)
    if total_error > 1e-10:
        print('bad special vectors. total_error=', total_error)
        print('errors=', errors)


# Randomized SVD (not sure where else to put this function)

def randomized_SVD(apply_A, apply_At, A_shape, min_rank=2, max_rank=20, oversampling_parameter=5, step=1, tol=1e-9,
                   verbose=True, check_consistency=False):
    nrow_A, ncol_A = A_shape
    def apply_A_ml(ww):
        return apply_A(*ww)

    Q = multilinear_randomized_range_finder(apply_A_ml, [ncol_A], min_rank=min_rank, max_rank=max_rank,
                                            oversampling_parameter=oversampling_parameter, step=step, tol=tol, verbose=True)
    R = np.zeros(Q.shape)
    for k in range(Q.shape[1]):
        R[:, k] = apply_At(Q[:, k].copy())  # Why is copy needed here?

    WL, ss, WR = np.linalg.svd(R.T, full_matrices=False)

    U = np.dot(Q, WL)
    ss = ss
    Vt = WR

    if check_consistency:
        A_dense = np.zeros([nrow_A, ncol_A])
        for k in range(A_dense.shape[1]):
            ek = np.zeros(A_dense.shape[1])
            ek[k] = 1.
            A_dense[:, k] = apply_A(ek)

        G = A_dense - np.dot(Q, np.dot(Q.T, A_dense))
        range_err = np.linalg.norm(G) / np.linalg.norm(A_dense)
        print('range_err=', range_err)

        svd_err = np.linalg.norm(np.dot(U, np.dot(np.diag(ss), Vt)) - A_dense) / np.linalg.norm(A_dense)
        print('svd_err=', svd_err)

        r = len(ss)
        true_U, true_ss, true_Vt = np.linalg.svd(A_dense, full_matrices=False)
        true_svd_err = np.linalg.norm(np.dot(true_U[:,:r], np.dot(np.diag(true_ss[:r]), true_Vt[:r,:])) - A_dense) / np.linalg.norm(A_dense)
        print('true_svd_err=', true_svd_err)

    return U, ss, Vt






