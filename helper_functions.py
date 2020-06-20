import numpy as np


def unit_vector(k, N):
    ek = np.zeros(N)
    ek[k] = 1.0
    return ek


