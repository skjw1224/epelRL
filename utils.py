"""
Various Functions used in several classes
"""
import os
import torch
import numpy as np

def scale(var, min, max, shift=True):  # [min, max] --> [-1, 1]
    shifting_factor = max + min if shift else torch.zeros_like(max)
    scaled_var = (2. * var - shifting_factor) / (max - min)
    return scaled_var

def descale(scaled_var, min, max):  # [-1, 1] --> [min, max]
    var = (max - min) / 2. * scaled_var + (max + min) / 2.
    return var

def reshape_wrapper(xvec, u, p_mu, p_sigma, p_eps, f):
    xmat = torch.reshape(xvec, [1, -1])
    return f(xmat, u, p_mu, p_sigma, p_eps)

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


# def derivs_discrete_time(fcn, fcn_type, dt, *args):
#     # Explicit RK temporal discretize --> Dormand Prince method
#     # can change this into 1st order
#
#     x, u, *p_args = args
#     device = t.device
#     C = torch.tensor([1. / 5., 3. / 10., 4. / 5., 8. / 9., 1.], device=device)
#     A = [torch.tensor([1. / 5.], device=device),
#          torch.tensor([3. / 40., 9. / 40.], device=device),
#          torch.tensor([44. / 45., -56. / 15., 32. / 9.], device=device),
#          torch.tensor([19372. / 6561., -25360. / 2187., 64448. / 6561., -212. / 729.], device=device),
#          torch.tensor([9017. / 3168., -355. / 33., 46732. / 5247., 49. / 176., -5103. / 18656.], device=device)]
#     B = torch.tensor([35. / 384., 0., 500. / 1113., 125. / 192., -2187. / 6784., 11. / 84.], device=device)
#     n_stages = len(C) + 1
#
#     s_dim = x.shape[-1]
#     a_dim = u.shape[-1]
#     p_dim = p_args[0].shape[-1]
#
#     # np.dot transpose [n, S, A(or S)] * [n, ] --> [S, A(or S), n] * [n, ] = [S, A(or S)]
#
#     if fcn_type == "dfdx":
#         dxp_mat = torch.zeros([n_stages, s_dim], device=device) # (n, s)
#         dfdxp_mat = torch.zeros([n_stages, s_dim, s_dim], device=device)
#         dxpdx_mat = torch.zeros([n_stages, s_dim, s_dim], device=device)
#         dfdx_mat = torch.zeros([n_stages, s_dim, s_dim], device=device)
#
#         dxp_mat[0], dfdxp_mat[0] = fcn(t, x, u, *p_args)
#         dxpdx_mat[0] = torch.eye(s_dim, device=device)
#         dfdx_mat[0] = dfdxp_mat[0] @ dxpdx_mat[0]
#
#         for n, (a, c) in enumerate(zip(A, C)):
#             # extract a row from dxp_mat into a 1D vector
#             dx = torch.mm(a.unsqueeze(0), dxp_mat[:n + 1]) * dt
#             dxp_mat[n + 1], dfdxp_mat[n + 1] = fcn(t + c * dt, x + dx, u, *p_args)
#             dxpdx_mat[n + 1] = torch.eye(s_dim) + dot_trans(dxpdx_mat[:n + 1, :], a) * dt
#             dfdx_mat[n + 1] = dfdxp_mat[n + 1] @ dxpdx_mat[n + 1]
#             delxp_delx = torch.eye(s_dim, device=device) + dot_trans(dfdx_mat, B) * dt
#             return delxp_delx
#
#     elif fcn_type in {"dfdu", "dfdp"}:
#         dxp_mat = torch.zeros([n_stages, s_dim], device=device)
#         if fcn_type == "dfdu":
#             dfdarg_mat = torch.zeros([n_stages, s_dim, a_dim], device=device)
#         else:
#             dfdarg_mat = torch.zeros([n_stages, s_dim, p_dim * 2], device=device)
#
#         dxp_mat[0], dfdarg_mat[0] = fcn(t, x, u, *p_args)
#         for n, (a, c) in enumerate(zip(A, C)):
#             dx = torch.mm(a.unsqueeze(0), dxp_mat[:n + 1]) * dt
#             dxp_mat[n + 1], dfdarg_mat[n + 1] = fcn(t + c * dt, x + dx, u, *p_args)
#         delxp_delarg = dot_trans(dfdarg_mat, B) * dt
#         return delxp_delarg
#
#     else:  # dcdx, dcTdx
#         delf_delx = fcn(x, u, *p_args)
#         return delf_delx

def dot_trans(x, y):
    """ xT @ y function to implement Dopri method

    :param x: Tensor<d0, d1, d2> to be transposed into <d1, d2, d0>
    :param y: Tensor<d0> another tensor involved in dot product
    """
    d = x.shape
    out = torch.empty([d[1], d[2]], device=x.device)
    for b in range(d[1]):
        for c in range(d[2]):
            out[b, c] = sum(xx * yy for xx, yy in zip(x[:, b, c], y))
    return out


def action_meshgen(single_dim_mesh, env_a_dim):
    n_grid = len(single_dim_mesh)
    single_dim_mesh = np.array(single_dim_mesh)
    a_dim = n_grid ** env_a_dim  # M ** A
    a_mesh = np.stack(np.meshgrid(*[single_dim_mesh for _ in range(env_a_dim)]))  # (A, M, M, .., M)
    a_mesh_idx = np.arange(a_dim).reshape(*[n_grid for _ in range(env_a_dim)])  # (M, M, .., M)

    return a_mesh, a_mesh_idx, a_dim

def action_idx2mesh(vec_idx, a_mesh, a_mesh_idx, a_dim):
    env_a_dim = len(a_mesh)

    mesh_idx = (a_mesh_idx == vec_idx).nonzero()
    a_nom = np.array([a_mesh[i, :][tuple(mesh_idx)] for i in range(env_a_dim)])
    return a_nom
