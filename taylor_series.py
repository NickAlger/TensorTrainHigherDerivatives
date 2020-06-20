import numpy as np
from math import factorial
from tensor_operations import tensor_train_symmetric_pushthrough
from tensor_train_from_tensor_action import tensor_train_from_tensor_action, randomized_SVD


def construct_derivative_tensor_approximations(QHD, apply_preconditioned_QHD, num_derivatives, rank):
    f0 = QHD.compute_derivative_of_quantity_of_interest([], None)[:]

    def apply_Jac(x):
        return apply_preconditioned_QHD([x], 1)

    def apply_Jac_t(x):
        return apply_preconditioned_QHD([x], 0)

    Jac_shape = (QHD.M.dim(), QHD.Z.dim())

    Jac_U, Jac_ss, Jac_Vt = randomized_SVD(apply_Jac, apply_Jac_t, Jac_shape, max_rank=rank, check_consistency=True)

    all_CC = []
    for k in range(2, num_derivatives+1):
        print('k=', k)

        tensor_shape = [QHD.M.dim() for _ in range(k)] + [QHD.Z.dim()]
        CC = tensor_train_from_tensor_action(apply_preconditioned_QHD, tensor_shape, max_rank=rank, verbose=True)

        all_CC.append(CC)

    return f0, Jac_U, Jac_ss, Jac_Vt, all_CC


def eval_taylor_series(dm, num_derivatives, derivative_tensor_pushthrough):
    # f(m + dm) =approx= f(m) + f'(m) dm + 1/2 f''(m) dm^2 + ... + 1/k! f^(k)(m) dm^k
    #
    # derivative_tensor_pushthrough([dm1, dm2, ..., dmi]) = f^(i)(m) dm1 dm2 ... dmi
    #
    # ff[0] = f(m)
    # ff[1] = f(m) + f'(m) dm
    # ...
    # ff[k] = f(m) + f'(m) dm + 1/2 f''(m) dm^2 + ... + 1/k! f^(k)(m) dm^k
    tt = []
    for k in range(num_derivatives+1):
        tt.append((1. / factorial(k)) * derivative_tensor_pushthrough([dm for _ in range(k)]))
    tt = np.vstack(tt).T
    ff = np.cumsum(tt, axis=1)
    return ff


def eval_tensor_train_taylor_series(dm, f0, Jac_U, Jac_ss, Jac_Vt, all_CC):
    def derivative_tensor_pushthrough(all_dm):
        k = len(all_dm)
        if k == 0:
            return f0
        elif k == 1:
            return np.dot(Jac_U, np.dot(np.diag(Jac_ss), np.dot(Jac_Vt, dm)))
        else:
            return tensor_train_symmetric_pushthrough(all_CC[k - 2], dm)

    num_derivatives = len(all_CC) + 1
    return eval_taylor_series(dm, num_derivatives, derivative_tensor_pushthrough)
