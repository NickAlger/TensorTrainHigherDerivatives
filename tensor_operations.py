import numpy as np


def dense_tensor_action(T, xx, k):
    # Computes T(xx[0], ..., xx[k-1],  .  , xx[k], ..., xx[-1])
    # (k'th mode of T is the output, all other modes are inputs)
    d = len(T.shape)
    X = T.copy()
    before_inds = list(range(k))
    after_inds = list(range(k, len(xx)))

    for ii in before_inds:
        v = xx[ii]
        X = np.dot(v, X.reshape((len(v), -1)))

    for jj in after_inds[::-1]:
        v = xx[jj]
        X = np.dot(X.reshape((-1, len(v))), v)

    return X.reshape(-1)


def build_dense_tensor_from_tensor_train(CC):
    T = np.array([1])
    for C in CC:
        r_left, N, r_right = C.shape
        T = np.dot(T, C.reshape((r_left, N*r_right))).reshape((-1, r_right))
    return T.reshape([C.shape[1] for C in CC])


def third_order_tensor_double_contraction(C, u, v, w):
    rL, n, rR = C.shape
    if u is None:
        return np.dot(np.dot(C.reshape((rL*n, rR)), w).reshape(rL, n), v).reshape(-1)
    elif v is None:
        return np.dot(u, np.dot(C.reshape((rL*n, rR)), w).reshape(rL, n)).reshape(-1)
    elif w is None:
        return np.dot(v, np.dot(u, C.reshape(rL, n*rR)).reshape(n, rR)).reshape(-1)
    else:
        raise RuntimeError('exactly one mode must be None')


def partial_tensor_train_pushthrough(CC, xx):
    v = np.array([1])
    for k in range(len(xx)):
        v = third_order_tensor_double_contraction(CC[k], v, xx[k], None)
    return v


def partial_tensor_train_pullback(CC, xx):
    v = np.array([1])
    for k in range(len(xx)):
        v = third_order_tensor_double_contraction(CC[-(k + 1)], None, xx[-(k + 1)], v)
    return v


def tensor_train_pushthrough(CC, xx):
    # [x1, x2, ..., xk] -> T(x1, x2, ..., xk, . )
    v = partial_tensor_train_pushthrough(CC, xx)
    last_core = CC[-1]
    return third_order_tensor_double_contraction(last_core, v, None, np.array([1])).reshape(-1)


def tensor_train_symmetric_pushthrough(CC, x):
    # x -> T(x, x, ..., x, . )
    xx = [x for _ in range(len(CC)-1)]
    return tensor_train_pushthrough(CC, xx)


def apply_tensor_train_sweeping(CC, xx_first, xx_last):
    # Computes T(xx_first[0], ..., xx_first[-1], . , xx_last[0], ..., xx_last[-1])
    q_left = partial_tensor_train_pushthrough(CC, xx_first)
    q_right = partial_tensor_train_pullback(CC, xx_last)
    m = len(xx_first)
    return third_order_tensor_double_contraction(CC[m], q_left, None, q_right)


def tensor_train_symmetric_pullback(CC, x, z, symmetrize=True):
    # symmetrize==False:
    #   x -> T( . , x, ..., x, z)
    # symmetrize==True:
    #   x -> T_sym( . , x, ..., x, z)
    # where
    #   T_sym( . , x, ..., x, z) := (1/k) * ( T( . , x, ..., x, z)
    #                                       + T(x, . , x, ..., x, z)
    #                                       + ...
    #                                       + T(x, x, ..., x, . , z) )
    num_derivatives = len(CC) - 1
    if symmetrize:
        ww = []
        for k in range(num_derivatives):
            xx_first = [x for _ in range(k)]
            xx_last = [x for _ in range(k+1,num_derivatives)] + [z]
            ww.append(apply_tensor_train_sweeping(CC, xx_first, xx_last))
        w = np.mean(ww, axis=0)
    else:
        xx_last = [x for _ in range(1, num_derivatives)] + [z]
        w = apply_tensor_train_sweeping(CC, [], xx_last)
    return w


def low_rank_factorization(X, max_rank, tol):
    [U, ss, VT] = np.linalg.svd(X, full_matrices=False)
    r0 = len(np.where(ss > tol)[0])
    r = np.min([r0, max_rank])
    C = U[:,:r]
    Z = np.dot(np.diag(ss[:r]), VT[:r,:])
    return C, Z

def make_tensor_train_with_conventional_method(T, max_rank=10, tol=1e-4):
    d = len(T.shape)
    X = np.reshape(T, (T.shape[0], -1))
    C1, X = low_rank_factorization(X, max_rank, tol)
    C1 = C1.reshape([1, C1.shape[0], C1.shape[1]])
    r_right = C1.shape[2]
    CC = [C1]
    for kk in range(1,d-1):
        r_left = r_right
        n = T.shape[kk]
        X = np.reshape(X, (r_left*n,-1))
        Ck, X = low_rank_factorization(X, max_rank, tol)
        r_right = Ck.shape[1]
        CC.append(np.reshape(Ck, (r_left, n, r_right)))
        X = X.reshape((X.shape[0], X.shape[1], 1))
    CC.append(X)
    return CC