import torch
import torch.nn as nn


class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        input_dim: size of each input sample
        output_dim: size of each output sample
    Shape:
        - Input: (N, input_dim) where N is an arbitrary batch size
        - Output: (N, output_dim) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (output_dim, input_dim).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        shape_params: the learnable scaling factors of shape (output_dim).
            The values are initialised as ones.

        basis_func_type: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, input_dim, output_dim, basis_func_type):
        super(RBF, self).__init__()
        # Dimension
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Parameters
        self.centres = nn.Parameter(torch.zeros(self.output_dim, self.input_dim))
        self.shape_params = nn.Parameter(torch.zeros(self.output_dim))
        self.reset_parameters()
        # Basis function
        self.basis_func_dict()
        self.basis_func = self.bases[basis_func_type]

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.shape_params, 1)

    def forward(self, x):  # (B, x)
        size = (x.size(0), self.output_dim, self.input_dim)  # (B, y, x)
        x = x.unsqueeze(1).expand(size)   # (B, y, x)
        c = self.centres.unsqueeze(0).expand(size)  # (B, y, x)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.shape_params.unsqueeze(0)  # (B, y)
        phi = self.basis_func(distances)

        return phi

    def basis_func_dict(self):
        self.bases = {
            'gaussian': self._gaussian,
            'linear': self._linear,
            'quadratic': self._quadratic,
            'inverse quadratic': self._inverse_quadratic,
            'multiquadric': self._multiquadric,
            'inverse multiquadric': self._inverse_multiquadric,
            'spline': self._spline,
            'poisson one': self._poisson_one,
            'poisson two': self._poisson_two,
            'matern32': self._matern32,
            'matern52': self._matern52
        }

    def _gaussian(self, alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi

    def _linear(self, alpha):
        phi = alpha
        return phi

    def _quadratic(self, alpha):
        phi = alpha.pow(2)
        return phi

    def _inverse_quadratic(self, alpha):
        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
        return phi

    def _multiquadric(self, alpha):
        phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
        return phi

    def _inverse_multiquadric(self, alpha):
        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
        return phi

    def _spline(self, alpha):
        phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
        return phi

    def _poisson_one(self, alpha):
        phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
        return phi

    def _poisson_two(self, alpha):
        phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) \
              * alpha * torch.exp(-alpha)
        return phi

    def _matern32(self, alpha):
        phi = (torch.ones_like(alpha) + 3 ** 0.5 * alpha) * torch.exp(-3 ** 0.5 * alpha)
        return phi

    def _matern52(self, alpha):
        phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) \
               * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
        return phi

