# Copyright (c) 2022-2023 Johnson Lee. All rights reserved.

# This file contains custom cnns which can specify layers.
# Pay attention to class name conflicts, e.g., `ResNet` in `torchvision.models`.


from torch import Tensor, nn
from torch.nn import functional as F

from .feature_extractor import (
    DenseNetFeatureExtractor,
    LeNetFeatureExtractor,
    ResNetFeatureExtractor,
)


class LeNet(LeNetFeatureExtractor):
    def forward(self, x) -> Tensor:
        return self.fc(F.sigmoid(self.net(x)))


class ResNet(ResNetFeatureExtractor):
    r"""Custom ResNet model.

    Args:
        num_layer: Number of layers in the ResNet model.
        pretrained: If True, returns a model pre-trained on ImageNet.
        **kwargs: Parameters passed to the :class:`torchvision.models.ResNet`
        base class. You can set `num_classes` here.

    Attributes:
        net: The complete :class:`torchvision.models.ResNet` model.
    """

    def __init__(self, num_layer: int, pretrained: bool = False, **kwargs):
        # Call resnet18(weights=ResNet18_Weights.DEFAULT, num_classes=10) yields error,
        # so remove it here and set fc layer below
        # `default_num_classes = self.net.fc.out_features` yields
        # `AttributeError: 'ResNet' object has no attribute 'net'` because `super().__init__()`
        # is not called yet, so we have to hardcode the default number of classes
        default_num_classes = 1000
        num_classes = kwargs.pop("num_classes", default_num_classes)

        super().__init__(num_layer, pretrained, **kwargs)
        self._compute_grad_on_fc()
        if num_classes != default_num_classes:
            # Avoid duplicate training of fc layer
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DenseNet(DenseNetFeatureExtractor):
    r"""Custom DenseNet model.

    Args:
        num_layer: Number of layers in the DenseNet model.
        pretrained: If True, returns a model pre-trained on ImageNet.
        **kwargs: Parameters passed to the :class:`torchvision.models.DenseNet`
        base class. You can set `num_classes` here.

    Attributes:
        net: The complete :class:`torchvision.models.DenseNet` model.
    """

    def __init__(self, num_layer: int, pretrained: bool = False, **kwargs):
        # Call densenet121(weights=DenseNet121_Weights.DEFAULT, num_classes=10) yields error,
        # so remove it here and set fc layer below
        default_num_classes = 1000
        num_classes = kwargs.pop("num_classes", default_num_classes)

        super().__init__(num_layer, pretrained, **kwargs)
        self._compute_grad_on_classifier()
        if num_classes != default_num_classes:
            # Avoid duplicate training of fc layer
            self.net.classifier = nn.Linear(
                self.net.classifier.in_features, num_classes
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
