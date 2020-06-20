import numpy as np
from fenics import *
from poisson_problem import nonlinear_neumann_poisson_problem
from tensor_train_from_tensor_action import tensor_train_from_tensor_action, randomized_SVD
from tensor_maximum_singular_value import tensor_maximum_singular_value
from tensor_operations import tensor_train_symmetric_pushthrough, tensor_train_symmetric_pullback
import scipy.sparse.linalg as spla
import pickle
from time import time
from multiprocessing import Pool
import sys

## WARNING: This is extremely expensive and takes a looong time to run.

nproc = 8

max_rank = 40
min_rank = 2
tol = 1e-3
mesh_nn = [10,20,30,40,50,60,70,80]
all_num_derivatives = [1,2,3,4,5,6]
all_tols = [1e-2, 1e-3]
generate_data = True
save_data = True

filename_base = 'poisson_mesh_scalability' + '_k' + str(np.max(all_num_derivatives))
results_filename = filename_base + '.p'

all_ranks = np.arange(min_rank, max_rank+1)


def one_run(mesh_n):
    log_filename = filename_base + '_n' + str(mesh_n) + '.out'
    print('starting '+ log_filename)
    old_stdout = sys.stdout
    sys.stdout = myoutput = open(log_filename, 'w')

    tensor_norms = np.zeros(len(all_num_derivatives))
    relative_errors = np.zeros((len(all_ranks), len(all_num_derivatives)))

    print('mesh_n=', mesh_n)
    Q, a, m, u, bcs, apply_sqrtC, apply_sqrtC_T, QHD, apply_preconditioned_QHD, forward_map = nonlinear_neumann_poisson_problem(mesh_n)

    for jj, num_derivatives in enumerate(all_num_derivatives):
        print('num_derivatives=', num_derivatives)
        tensor_shape = [QHD.M.dim() for _ in range(num_derivatives)] + [QHD.Z.dim()]

        if num_derivatives == 1:
            def apply_Jac(x):
                return apply_preconditioned_QHD([x], 1)


            def apply_Jac_t(x):
                return apply_preconditioned_QHD([x], 0)

            apply_Jac_linop = spla.LinearOperator((QHD.Z.dim(), QHD.M.dim()), matvec=apply_Jac, rmatvec=apply_Jac_t)
            tensor_norm = np.max(spla.svds(apply_Jac_linop, k=1, which='LM', return_singular_vectors=False))

        else:
            def T_pushthrough(x):
                return apply_preconditioned_QHD([x for _ in range(num_derivatives)], num_derivatives)

            def T_pullback(x,z):
                return apply_preconditioned_QHD([x for _ in range(num_derivatives-1)] + [z], 0)

            tensor_norm = tensor_maximum_singular_value(T_pushthrough, T_pullback, QHD.M.dim())

        print('tensor_norm=', tensor_norm, '\n')
        tensor_norms[jj] = tensor_norm

        for kk, rank in enumerate(all_ranks):
            print('rank=', rank)
            t = time()
            if num_derivatives == 1:
                Jac_shape = (QHD.M.dim(), QHD.Z.dim())

                Jac_U, Jac_ss, Jac_Vt = randomized_SVD(apply_Jac, apply_Jac_t, Jac_shape, max_rank=rank,
                                                       check_consistency=False)

                def apply_Jac_error(x):
                    y_true = apply_Jac(x)
                    y = np.dot(Jac_U, np.dot(np.diag(Jac_ss), np.dot(Jac_Vt, x)))
                    return y_true - y

                def apply_Jac_error_t(x):
                    y_true = apply_Jac_t(x)
                    y = np.dot(Jac_Vt.T, np.dot(np.diag(Jac_ss), np.dot(Jac_U.T, x)))
                    return y_true - y

                apply_Jac_error_linop = spla.LinearOperator((QHD.Z.dim(), QHD.M.dim()), matvec=apply_Jac_error, rmatvec=apply_Jac_error_t)

                error_norm = np.max(spla.svds(apply_Jac_error_linop, k=1, which='LM', return_singular_vectors=False))

            else:
                CC = tensor_train_from_tensor_action(apply_preconditioned_QHD, tensor_shape, max_rank=rank, tol=1e-9, verbose=True)

                def error_pushthrough(x):
                    return T_pushthrough(x) - tensor_train_symmetric_pushthrough(CC, x)

                def error_pullback(x,z):
                    return T_pullback(x,z) - tensor_train_symmetric_pullback(CC, x, z, symmetrize=True)

                error_norm = tensor_maximum_singular_value(error_pushthrough, error_pullback, QHD.M.dim())

            relative_error = error_norm / tensor_norm
            relative_errors[kk,jj] = relative_error
            time_elapsed = time() - t

            print('mesh_n=', mesh_n, ', num_derivatives=', num_derivatives, ', rank=', rank, ', relative_error=', relative_error)
            print('time_elapsed=', time_elapsed)
            print('')

    myoutput.close()
    sys.stdout = old_stdout
    return (relative_errors, tensor_norms)


if generate_data:
    with Pool(nproc) as pool:
        results = pool.map(one_run, mesh_nn)

    all_tensor_norms = np.zeros((len(all_num_derivatives), len(mesh_nn)))
    all_relative_errors = np.zeros((len(all_ranks), len(all_num_derivatives), len(mesh_nn)))
    for ii, mesh_n in enumerate(mesh_nn):
        all_relative_errors[:,:,ii] = results[ii][0]
        all_tensor_norms[:,ii] = results[ii][1]


    if save_data:
        data = (mesh_nn, all_num_derivatives, all_ranks, all_relative_errors, all_tensor_norms)
        pickle.dump(data, open(results_filename, 'wb'))
else:
    (mesh_nn, all_num_derivatives, all_ranks, all_relative_errors, all_tensor_norms) = pickle.load(open(results_filename, 'rb'))


ranks_achieving_tols = -1.0 * np.ones((len(all_tols), len(mesh_nn), len(all_num_derivatives)))
for ll, tol in enumerate(all_tols):
    for ii, _ in enumerate(mesh_nn):
        for jj, _ in enumerate(all_num_derivatives):
            successful_inds = np.where(all_relative_errors[:,jj,ii] < tol)[0]
            if len(successful_inds) > 0:
                ranks_achieving_tols[ll,ii,jj] = all_ranks[np.min(successful_inds)]


print('mesh_nn=', mesh_nn)
print('')
print('all_num_derivatives=', all_num_derivatives)
for ll in range(len(all_tols)):
    print('')
    print('all_tols[ll]=', all_tols[ll])
    print('ranks_achieving_tol[ll,:,:]=\n', ranks_achieving_tols[ll,:,:])

