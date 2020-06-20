import numpy as np
from hilbert_tensor import make_hilbert_tensor
from tensor_train_from_tensor_action import tensor_train_from_tensor_action
from tensor_operations import dense_tensor_action, make_tensor_train_with_conventional_method, build_dense_tensor_from_tensor_train
import scipy.io as sio
import pickle


generate_data=True
save_data = True
make_plot=True
save_plot=True
d = 5

filename = 'hilbert_convergence.p'

if generate_data:
    T_shape = 41 + np.arange(d) # e.g., [41, 42, 43, 44, 45]
    T = make_hilbert_tensor(T_shape)
    apply_T = lambda xx, k: dense_tensor_action(T, xx, k)
    rr = 2+np.arange(18)
    errs = np.zeros(len(rr))
    errs_optimal = np.zeros(len(rr))
    for k in range(len(rr)):
        r = rr[k]
        CC = tensor_train_from_tensor_action(apply_T, T_shape, max_rank=r, tol=1e-20, verbose=True)

        T2 = build_dense_tensor_from_tensor_train(CC)
        tt_err = np.linalg.norm(T2 - T) / np.linalg.norm(T)
        errs[k] = tt_err

        CC_optimal = make_tensor_train_with_conventional_method(T, tol=1e-20, max_rank=r)

        T_optimal = build_dense_tensor_from_tensor_train(CC_optimal)

        tt_optimal_err = np.linalg.norm(T_optimal - T) / np.linalg.norm(T)
        errs_optimal[k] = tt_optimal_err

        print('r=', r, ', tt_err=', tt_err, ', tt_optimal_err=', tt_optimal_err)

    if save_data:
        data = (T_shape, rr, errs, errs_optimal)
        pickle.dump(data, open(filename, 'wb'))
else:
    (T_shape, rr, errs, errs_optimal) = pickle.load(open(filename, 'rb'))

if make_plot:
    import matplotlib.pyplot as plt

    plt.figure()
    if d == 5:
        tt_cross_data = sio.loadmat('hilbert_dmrg_cross_data2.mat') # from nick_hilbert_tensor_train_cross.m, using Oseledets TT-cross matlab code
        errs_cross = tt_cross_data['errs'].reshape(-1)
        rr_cross = tt_cross_data['rr'].reshape(-1)
        plt.semilogy(rr_cross, errs_cross,linestyle='dashed', color='k')

    plt.semilogy(rr.reshape(-1), errs.reshape(-1), color='k')
    plt.semilogy(rr.reshape(-1), errs_optimal.reshape(-1),linestyle='dotted', color='k')
    plt.title('Hilbert tensor: approximation error vs. rank',fontsize=14)
    plt.xlabel('Tensor train rank, $r$',fontsize=12)
    plt.ylabel(r'$\frac{||T - \widetilde{T}||}{||T||}$', rotation=0, fontsize=16, labelpad=30)
    plt.xlim([2,19])
    if d == 5:
        plt.legend(['TT-cross', 'TT-RSVD (our method)', 'TT-SVD (optimal)'])
    else:
        plt.legend(['TT-RSVD (our method)', 'TT-SVD (optimal)'])

if make_plot and save_plot:
    plt.savefig("hilbert_test1.pdf",bbox_inches='tight',dpi=100)


