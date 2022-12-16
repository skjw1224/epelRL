"""
Various Functions used in several classes
"""

import torch

def scale(var, min, max, shift=True):  # [min, max] --> [-1, 1]
    shifting_factor = max + min if shift else torch.zeros_like(max)
    scaled_var = (2. * var - shifting_factor) / (max - min)
    return scaled_var

def descale(scaled_var, min, max):  # [-1, 1] --> [min, max]
    var = (max - min) / 2. * scaled_var + (max + min) / 2.
    return var

def reshape_wrapper(t, xvec, u, p_mu, p_sigma, p_eps, f):
    xmat = torch.reshape(xvec, [1, -1])
    return f(t, xmat, u, p_mu, p_sigma, p_eps)

def jacobian(f, x):
    """Compute the jacobian of f(x) w.r.t. x_index

        Args:
            f (Tensor<1, n> -> Tensor<1, m>): Function to differentiate.
            index ([ 1 ]) : the indices of xs in interest
            xs ( Tensor1<1, n1> .. Tensork<1, nk> )
                : k different Tensors to differentiate with respect to.

            where m (int): Expected output dimensions.
                ni (int) : size of the i th argument

        Returns:
            dfdx ( Tensor<m, n> } : Jacobian-matrix
    """

    dfdx = [grad(f[0, mi], x)[0] for mi in range(f.shape[-1])]
    dfdx = torch.stack(dfdx)
    dfdx.requires_grad_()

    return dfdx

def grad(y, x, allow_unused=True):
    """Evaluates the gradient of y w.r.t x safely.

        Args:
            y (Tensor<0>): Tensor to differentiate.
            x (Tensor<n>): Tensor to differentiate with respect to.
            **kwargs: Additional key-word arguments to pass to `torch.autograd.grad()`.

        Returns:
            Gradient (Tensor<n>).
        """
    dy_dx, = torch.autograd.grad(
        y, x, create_graph=True, allow_unused=allow_unused)

    # The gradient is None if disconnected.
    dy_dx = dy_dx if dy_dx is not None else torch.zeros_like(x)
    dy_dx.requires_grad_()

    return dy_dx

# def hessian(fs, xs, ys=None):
#     """ Batch Hessian computing function: needed for a LQI controller
#
#         Args:
#             fs <N,1> vector of y(scalar function value w.r.t. xs)
#             xs <N,n> input variable vector; N, batch size; n, input variable size
#             ys <N,m> second input variable vector; N, batch size; m, input variable size (optional)
#
#         Return:
#             hess <N,n,m> Hessian matrix, 2nd derivative of y w.r.t. (xs, ys)
#     """
#     if ys is None:
#         ys = xs
#
#     n = xs.size(-1) # xs dim
#     m = ys.size(-1) # ys dim
#     jac = torch.autograd.grad(fs, xs, create_graph=True)[0] # (1, n)
#     hess = torch.zeros([n, m])
#     for ii in range(m):
#         v = torch.zeros([1, m])
#         v[0, ii] = 1.
#         hess[ii, :] = torch.autograd.grad(jac, ys, v, retain_graph=True)[0]
#     # for N, x in enumerate(xs):
#     #     g2 = torch.autograd.jac(jac, xs, torch.eye(xs.size(-1)), retain_graph=True)[0]
#     #     if N == 0:
#     #         hess = g2
#     #     else:
#     #         hess = torch.cat([hess, g2])
#     return hess
