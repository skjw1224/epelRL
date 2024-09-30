import numpy as np


class RBF(object):
    def __init__(self, input_dim, output_dim, basis_func_type):
        # Dimension
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Parameters
        self.centers = np.random.randn(self.output_dim, self.input_dim)
        self.shape_params = np.ones((self.output_dim))
        # Basis function
        self._set_basis_function(basis_func_type)

    def forward(self, inputs):  # (B, X)
        shape = (inputs.shape[0], self.output_dim, self.input_dim)  # (B, F, X)
        inputs = np.broadcast_to(np.expand_dims(inputs, axis=1), shape=shape)
        centers = np.broadcast_to(np.expand_dims(self.centers, axis=0), shape=shape)
        shape_params = np.broadcast_to(np.expand_dims(self.shape_params, axis=0), shape=shape[:2])
        distances = np.linalg.norm(inputs - centers, axis=2) * shape_params
        phi = self.basis_function(distances)  # (B, F)

        return phi

    def _set_basis_function(self, basis_func_type):
        basis_func_dict = {
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
        self.basis_function = basis_func_dict[basis_func_type]

    def _gaussian(self, alpha):
        phi = np.exp(-0.1 * alpha ** 2)
        return phi

    def _linear(self, alpha):
        phi = alpha
        return phi

    def _quadratic(self, alpha):
        phi = alpha ** 2
        return phi

    def _inverse_quadratic(self, alpha):
        phi = np.ones_like(alpha) / (np.ones_like(alpha) + alpha**(2))
        return phi

    def _multiquadric(self, alpha):
        phi = (np.ones_like(alpha) + alpha**(2))**(0.5)
        return phi

    def _inverse_multiquadric(self, alpha):
        phi = np.ones_like(alpha) / (np.ones_like(alpha) + alpha**(2))**(0.5)
        return phi

    def _spline(self, alpha):
        phi = (alpha**(2) * np.log(alpha + np.ones_like(alpha)))
        return phi

    def _poisson_one(self, alpha):
        phi = (alpha - np.ones_like(alpha)) * np.exp(-alpha)
        return phi

    def _poisson_two(self, alpha):
        phi = ((alpha - 2 * np.ones_like(alpha)) / 2 * np.ones_like(alpha)) \
              * alpha * np.exp(-alpha)
        return phi

    def _matern32(self, alpha):
        phi = (np.ones_like(alpha) + 3 ** 0.5 * alpha) * np.exp(-3 ** 0.5 * alpha)
        return phi

    def _matern52(self, alpha):
        phi = (np.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) \
               * alpha**(2)) * np.exp(-5 ** 0.5 * alpha)
        return phi
