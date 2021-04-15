"""Extensions to the models module."""
import collections
from typing import Any

from torch import nn
from torchvision import models


def __getattr__(name: str) -> Any:
    """Forward to `torchvision.models`."""
    return getattr(models, name)


def resnet18_seq(**kwargs: Any) -> nn.Sequential:
    """Return sequentialized ResNet-18 model from torchvision."""
    model = models.resnet18(**kwargs)
    return nn.Sequential(
        collections.OrderedDict((
            ('conv1', model.conv1),
            ('bn1', model.bn1),
            ('relu', model.relu),
            ('maxpool', model.maxpool),
            ('layer1', model.layer1),
            ('layer2', model.layer2),
            ('layer3', model.layer3),
            ('layer4', model.layer4),
            ('avgpool', model.avgpool),
            ('flatten', nn.Flatten()),
            ('fc', model.fc),
        )))
