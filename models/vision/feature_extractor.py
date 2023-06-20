# Copyright (c) 2022-2023 Johnson Lee. All rights reserved.

# This file is a collection of feature extractors for vision tasks, i.e.,
# output does not go through the classifier layer.
# Can be used as the feature extractor of DKL Model.


import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import *
from transformers import ViTConfig, ViTModel


class LeNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
        )
        self.fc = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, num_layer: int, pretrained: bool = False, **kwargs):
        super().__init__()
        assert num_layer in [
            18,
            34,
            50,
            101,
            152,
        ], "num_layer must be in [18, 34, 50, 101, 152]"
        # Also can use `getattr(torchvision.models, f"resnet{num_layer}")` here
        net = eval(f"resnet{num_layer}")
        # Also can use
        # `operator.attrgetter(f"ResNet{num_layer}_Weights.DEFAULT")(torchvision.models)` here
        weights = eval(f"ResNet{num_layer}_Weights.DEFAULT")
        self.net = net(weights=weights, **kwargs) if pretrained else net(**kwargs)
        self._compute_grad_on_fc(False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    # Because in feature extractor, fc layer is not use in the forward pass,
    # so we need to disable the gradient computation on fc layer
    # when wrapped the feature extractor in `DistrubutedDataParallel` to avoid `RuntimeError`:
    """
    Traceback (most recent call last):
    ...
    File "~/anaconda3/envs/torch/lib/python3.11/site-packages/torch/multiprocessing/spawn.py", line 239, in spawn
        return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "~/anaconda3/envs/torch/lib/python3.11/site-packages/torch/multiprocessing/spawn.py", line 197, in start_processes
        while not context.join():
                ^^^^^^^^^^^^^^
    File "~/anaconda3/envs/torch/lib/python3.11/site-packages/torch/multiprocessing/spawn.py", line 160, in join
        raise ProcessRaisedException(msg, error_index, failed_process.pid)
    torch.multiprocessing.spawn.ProcessRaisedException: 

    -- Process 1 terminated with the following error:
    Traceback (most recent call last):
    File "~/anaconda3/envs/torch/lib/python3.11/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap
        fn(i, *args)
    ...
    File "~/anaconda3/envs/torch/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
        return forward_call(*args, **kwargs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "~/anaconda3/envs/torch/lib/python3.11/site-packages/torch/nn/parallel/distributed.py", line 1139, in forward
        if torch.is_grad_enabled() and self.reducer._rebuild_buckets():
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
    This error indicates that your module has parameters that were not used in producing loss.
    You can enable unused parameter detection by passing the keyword argument
    `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by 
    making sure all `forward` function outputs participate in calculating loss.

    If you already have done the above,
    then the distributed data parallel module wasn't able to locate the output tensors
    in the return value of your module's `forward` function.
    Please include the loss function and the structure of
    the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).

    Parameters which did not receive grad for rank 1:
    feature_extractor.net.fc.bias, feature_extractor.net.fc.weight

    Parameter indices which did not receive grad for rank 1: 60 61
    """

    def _compute_grad_on_fc(self, compute: bool = True):
        for param in self.net.fc.parameters():
            param.requires_grad = compute


class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, num_layer: int, pretrained: bool = False, **kwargs):
        super().__init__()
        assert num_layer in [
            121,
            161,
            169,
            201,
        ], "num_layer must be in [121, 161, 169, 201]"
        # Also can use `getattr(torchvision.models, f"densenet{num_layer}")` here
        net = eval(f"densenet{num_layer}")
        # Also can use
        # `operator.attrgetter(f"DenseNet{num_layer}_Weights.DEFAULT")(torchvision.models)` here
        weights = eval(f"DenseNet{num_layer}_Weights.DEFAULT")
        self.net = net(weights=weights, **kwargs) if pretrained else net(**kwargs)
        self._compute_grad_on_classifier(False)

    def forward(self, x: Tensor) -> Tensor:
        features = self.net.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def _compute_grad_on_classifier(self, compute: bool = True):
        for param in self.net.classifier.parameters():
            param.requires_grad = compute


class HuggingFaceViTFeatureExtractor(nn.Module):
    def __init__(self, pretrained_model_name: str = None):
        super().__init__()
        self.vit = (
            ViTModel.from_pretrained(pretrained_model_name)
            if pretrained_model_name
            else ViTModel(ViTConfig())
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.vit(x).pooler_output

    def get_hidden_size(self) -> int:
        return self.vit.config.hidden_size

    # No need to disable the gradient computation on classifier layer
    # because `ViTModel` is a pure feature extractor, i.e.,
    # it does not have a classifier layer
