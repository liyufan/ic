import math

import gpytorch
import torch
from torch import Tensor


# https://github.com/cornellius-gp/gpytorch/blob/master/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.ipynb
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        num_dim: int,
        grid_bounds: tuple = (-10.0, 10.0),
        grid_size: int = 64,
        multi_device: bool = False,
    ):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.GridInterpolationVariationalStrategy(
                    self,
                    grid_size=grid_size,
                    grid_bounds=[grid_bounds],
                    variational_distribution=variational_distribution,
                ),
                num_tasks=num_dim,
            )
        )
        super().__init__(variational_strategy)

        base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        device_ids = [
            torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
        ]

        self.covar_module = (
            gpytorch.kernels.MultiDeviceKernel(
                base_kernel=base_covar_module, device_ids=device_ids
            )
            if multi_device
            else base_covar_module
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x: Tensor):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
