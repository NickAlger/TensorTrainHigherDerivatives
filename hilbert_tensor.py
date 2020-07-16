import numpy as np

run_test = False

def make_hilbert_tensor(shape):
    # T_ijkl = 1/(i+j+k+l)
    kk = np.unravel_index(np.arange(np.prod(shape)), shape)
    T = np.zeros(shape)
    T[kk] = 1. / np.sum(tuple(np.array(kk) + 1), axis=0)
    return T


if run_test:
    shape = (3,2,5,4)
    T_true = np.zeros(shape)
    for ii in [1,2,3]:
        for jj in [1,2]:
            for kk in [1,2,3,4,5]:
                for ll in [1,2,3,4]:
                    T_true[ii-1, jj-1, kk-1, ll-1] = 1. / (ii+jj+kk+ll)

    T = make_hilbert_tensor(shape)

    err_make_hilbert_tensor = np.linalg.norm(T - T_true)/np.linalg.norm(T_true)
    print('err_make_hilbert_tensor=', err_make_hilbert_tensor)

    S = make_hilbert_tensor((3,3))
    S_true = np.array([[1./(1.+1.), 1./(1.+2.), 1./(1.+3.)],
                       [1./(2.+1.), 1./(2.+2.), 1./(2.+3.)],
                       [1./(3.+1.), 1./(3.+2.), 1./(3.+3.)]])
    err_3x3 = np.linalg.norm(S_true - S)/np.linalg.norm(S_true)
    print('err_3x3=', err_3x3)