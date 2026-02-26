from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn


def layer_init(layer: nn.Module, std: float = sqrt(2.0), bias_const: float = 0.0) -> nn.Module:
    if hasattr(layer, "weight") and layer.weight is not None:
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
