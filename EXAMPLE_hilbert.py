import numpy as np
from hilbert_tensor import make_hilbert_tensor
from tensor_operations import dense_tensor_action, build_dense_tensor_from_tensor_train
from tensor_train_from_tensor_action import *


# Demonstration of how to use tensor_train_from_tensor_action() to compress a tensor

tensor_shape = (21,22,23,24,25)
T = make_hilbert_tensor(tensor_shape)
T_action = lambda xx, k: dense_tensor_action(T, xx, k)

CC = tensor_train_from_tensor_action(T_action, tensor_shape, tol=1e-6, verbose=True, max_rank=20, step=3)

T2 = build_dense_tensor_from_tensor_train(CC)
error = np.linalg.norm(T2 - T)/np.linalg.norm(T)
print('error=', error)
