import numpy as np
from hilbert_tensor import make_hilbert_tensor
from tensor_operations import make_tensor_train_with_conventional_method, build_dense_tensor_from_tensor_train
import matplotlib.pyplot as plt

d = 5

# T_shape = [21,22,23,24,25]
# T_shape = [31,32,33,34,35]
T_shape = [41,42,43,44,45]
T = make_hilbert_tensor(T_shape)
rr = 2+np.arange(18)
errs = np.zeros(len(rr))
for k in range(len(rr)):
    r = rr[k]
    CC = make_tensor_train_with_conventional_method(T, tol=1e-20, max_rank=r)
    T2 = build_dense_tensor_from_tensor_train(CC)
    tt_err = np.linalg.norm(T2 - T) / np.linalg.norm(T)
    errs[k] = tt_err
    print('r=', r, ', tt_err=', tt_err,)


plt.figure()
plt.semilogy(rr.reshape(-1), errs.reshape(-1), color='k')
plt.legend(['TT-RSVD'])
plt.title('Hilbert tensor: approximation error vs. rank',fontsize=14)
plt.xlabel('Tensor train rank, $r$',fontsize=12)
plt.ylabel(r'$\frac{||T - \widetilde{T}||}{||T||}$', rotation=0, fontsize=16, labelpad=30)
plt.xlim([2,19])
