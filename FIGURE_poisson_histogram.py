import numpy as np
from fenics import *
from poisson_problem import nonlinear_neumann_poisson_problem
from taylor_series import construct_derivative_tensor_approximations, eval_tensor_train_taylor_series
import pickle


set_log_level(40)

mesh_n = 40
rank = 30
num_derivatives = 6
num_samples = 1000
generate_data = True
save_data = True
make_plot = True
save_plot = True

filename = 'poisson_histogram_6th.p'

if generate_data:
    Q, a, m, u, bcs, apply_sqrtC, apply_sqrtC_T, QHD, apply_preconditioned_QHD, forward_map = nonlinear_neumann_poisson_problem(mesh_n)

    f0, Jac_U, Jac_ss, Jac_Vt, all_CC = construct_derivative_tensor_approximations(QHD, apply_preconditioned_QHD, num_derivatives, rank)


    m0_vec = m.vector()[:].copy()

    ff_true = np.zeros([num_samples, QHD.M.dim(), 1])
    fff_train = np.zeros([num_samples, QHD.M.dim(), num_derivatives + 1])
    for k in range(num_samples):
        print('k=', k)
        dm_vec = np.random.randn(QHD.M.dim())

        ff_train = eval_tensor_train_taylor_series(dm_vec, f0, Jac_U, Jac_ss, Jac_Vt, all_CC)

        f_true = forward_map(m0_vec + apply_sqrtC(dm_vec))

        ff_true[k, :, 0] = f_true
        fff_train[k, :, :] = ff_train

    true_mean = np.mean(ff_true, axis=0)
    eee_train = ff_true - fff_train

    average_norm = np.mean(np.linalg.norm((f0.reshape((1,-1,1)) - ff_true), axis=1), axis=0).squeeze()
    all_error = np.linalg.norm(eee_train, axis=1)
    all_error_normalized = all_error / average_norm
    average_error = np.mean(all_error, axis=0)
    std_error = np.std(all_error, axis=0)

    normalized_mean_error = average_error / average_norm
    normalized_std_error = std_error / average_norm

    if save_data:
        data = (mesh_n, rank, num_derivatives, num_samples, normalized_mean_error, normalized_std_error, all_error_normalized)
        pickle.dump(data, open(filename, 'wb'))
else:
    (mesh_n, rank, num_derivatives, num_samples, normalized_mean_error, normalized_std_error, all_error_normalized) = pickle.load(open(filename, 'rb'))

    
print('normalized_mean_error=', normalized_mean_error)
print('normalized_std_error=', normalized_std_error)

if make_plot:
    import matplotlib.pyplot as plt

    nbins = 50
    logbins = np.geomspace(np.min(all_error_normalized), np.max(all_error_normalized), nbins)

    num_derivatives = all_error_normalized.shape[1] - 1
    fig, all_ax = plt.subplots(num_derivatives+1, sharex=True)
    for k in range(num_derivatives+1):
        ax = all_ax[k]
        ax.get_yaxis().set_ticks([])
        ax.hist(all_error_normalized[:, k], bins=logbins, color='gray')
        ordertitle = 'Order ' + str(k)
        ax.set_ylabel(ordertitle, rotation=0)
        ax.yaxis.set_label_position("right")
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.xscale('log')
    plt.xlabel('Normalized error')
    fig.subplots_adjust(top=0.93)
    fig.suptitle('Taylor series error histograms')
    fig.text(0.08, 0.5, 'Histogram count', va='center', rotation='vertical')

    if save_plot:
        plt.savefig('TaylorHistograms.pdf',
                    bbox_inches='tight',
                    pad_inches=0)

    plt.show()