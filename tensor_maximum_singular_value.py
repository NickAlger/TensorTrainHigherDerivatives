import numpy as np


def tensor_squared_power_iteration(symmetric_pushthrough, symmetric_pullback, in_dim,
                                   tol=1e-3, mu=0.25, maxiter=100, verbose=False):
    u = np.random.randn(in_dim)
    u = u/np.linalg.norm(u)

    z = symmetric_pushthrough(u)
    tensor_norm = np.linalg.norm(z)
    if verbose:
        print('initial tensor_norm=', tensor_norm)
    for k in range(maxiter):
        p = symmetric_pullback(u, z)
        u = (1.-mu) * (p/np.linalg.norm(p)) + mu * u
        u = u/np.linalg.norm(u)

        z = symmetric_pushthrough(u)
        tensor_norm2 = np.linalg.norm(z)
        err = np.abs(tensor_norm2 - tensor_norm) / np.abs(tensor_norm)
        tensor_norm = tensor_norm2
        if verbose:
            print('k=', k, ', tensor_norm=', tensor_norm)
        if err < tol:
            if verbose:
                print('tol achieved at k=', k)
            break

    return tensor_norm


def tensor_maximum_singular_value(symmetric_pushthrough, symmetric_pullback, in_dim, num_samples=5, verbose=True):
    ee = [tensor_squared_power_iteration(symmetric_pushthrough, symmetric_pullback, in_dim, verbose=verbose)
          for _ in range(num_samples)]
    if verbose:
        print('num_samples=', num_samples, ', np.max(ee)=', np.max(ee), ', np.min(ee)=', np.min(ee))
    return np.max(ee)