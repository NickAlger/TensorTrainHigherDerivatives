import numpy as np

def make_hilbert_tensor(shape):
    # T_ijkl = 1/(i+j+k+l)
    kk = np.unravel_index(np.arange(np.prod(shape)), shape)
    T = np.zeros(shape)
    T[kk] = 1. / (1. + np.sum(kk, axis=0))
    return T