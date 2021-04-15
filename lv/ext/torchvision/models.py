"""Extensions to the models module."""
import collections
from typing import Any

from torch import nn
from torchvision import models


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


def vgg16_seq(**kwargs: Any) -> nn.Sequential:
    """Return sequentialized VGG 16-layer."""
    model = models.vgg16(**kwargs)
    layers = list(
        zip([
            'conv1_1',
            'relu1_1',
            'conv1_2',
            'relu1_2',
            'pool1',
            'conv2_1',
            'relu2_1',
            'conv2_2',
            'relu2_2',
            'pool2',
            'conv3_1',
            'relu3_1',
            'conv3_2',
            'relu3_2',
            'conv3_3',
            'relu3_3',
            'pool3',
            'conv4_1',
            'relu4_1',
            'conv4_2',
            'relu4_2',
            'conv4_3',
            'relu4_3',
            'pool4',
            'conv5_1',
            'relu5_1',
            'conv5_2',
            'relu5_2',
            'conv5_3',
            'relu5_3',
            'pool5',
        ], model.features))
    layers += [('avgpool', model.avgpool), ('flatten', nn.Flatten())]
    layers += zip([
        'fc6',
        'relu6',
        'drop6',
        'fc7',
        'relu7',
        'drop7',
        'fc8a',
    ], model.classifier)
    return nn.Sequential(collections.OrderedDict(layers))
