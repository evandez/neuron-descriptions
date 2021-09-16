"""Extensions to the models module."""
import collections
import ssl
from typing import Any

from torch import nn
from torchvision import models

# Workaround for an annoying bug in torchvision...
ssl._create_default_https_context = ssl._create_unverified_context


def __getattr__(name: str) -> Any:
    """Forward to `torchvision.models`."""
    return getattr(models, name)


def alexnet_seq(**kwargs: Any) -> nn.Sequential:
    """Return sequentialized AlexNet model from torchvision."""
    model = models.alexnet(**kwargs)
    layers = list(
        zip([
            'conv1',
            'relu1',
            'pool1',
            'conv2',
            'relu2',
            'pool2',
            'conv3',
            'relu3',
            'conv4',
            'relu4',
            'conv5',
            'relu5',
            'pool5',
        ], model.features))
    layers += [('avgpool', model.avgpool), ('flatten', nn.Flatten())]
    layers += zip([
        'dropout6',
        'fc6',
        'relu6',
        'dropout7',
        'fc7',
        'relu7',
        'linear8',
    ], model.classifier)
    return nn.Sequential(collections.OrderedDict(layers))


def resnet18_seq(**kwargs: Any) -> nn.Sequential:
    """Return sequentialized ResNet-18 model from torchvision."""
    model = models.resnet18(**kwargs)
    return nn.Sequential(
        collections.OrderedDict([
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
        ]))


def resnet152_seq(**kwargs: Any) -> nn.Sequential:
    """Return sequentialized ResNet-152 model from torchvision."""
    model = models.resnet152(**kwargs)
    return nn.Sequential(
        collections.OrderedDict([
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
        ]))
