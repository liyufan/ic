import gpytorch
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from torch import Tensor, nn

from .layers import GaussianProcessLayer


class DKLModel(gpytorch.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        num_dim: int,
        grid_bounds: tuple = (-10.0, 10.0),
        multi_device: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(
            num_dim=num_dim, grid_bounds=grid_bounds, multi_device=multi_device
        )
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            self.grid_bounds[0], self.grid_bounds[1]
        )

    def forward(self, x: Tensor) -> MultitaskMultivariateNormal:
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res
