"""Simple extension to resnet152.py from NetDissect."""
import collections
from typing import Any

from third_party import resnet152

from torch import nn


def __getattr__(name):
    """Forward to original resnet152 source."""
    return getattr(resnet152, name)


class OldResNet152(nn.Sequential):
    """An old version of ResNet152 that we have Places365 weights for."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize ResNet152 and name the layers."""
        model = resnet152.OldResNet152(*args, **kwargs)
        children = collections.OrderedDict(
            zip([
                'conv1',
                'bn1',
                'relu',
                'maxpool',
                'layer1',
                'layer2',
                'layer3',
                'layer4',
                'avgpool',
                'flatten',
                'fc',
            ], model.children()))
        super().__init__(children)
