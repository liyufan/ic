# Copyright (c) 2022-2023 Johnson Lee. All rights reserved.


from torch import Tensor, nn
from transformers import ViTConfig, ViTForImageClassification


class HuggingFaceViT(nn.Module):
    def __init__(self, pretrained_model_name: str | None = None, **kwargs):
        r"""
        Args:
            pretrained_model_name: Name of the pretrained ViT.
            **kwargs: Parameters passed to the ``transformers.ViTConfig``.
            You can set ``num_labels`` (not `num_classes`!) here.
        """
        super().__init__()
        config = ViTConfig(**kwargs)
        if not pretrained_model_name:
            self.vit = ViTForImageClassification(config)
        else:
            self.vit = ViTForImageClassification.from_pretrained(
                pretrained_model_name, config=config, ignore_mismatched_sizes=True
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.vit(x).logits
