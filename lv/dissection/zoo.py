"""Defines dissection configurations."""
from typing import Any

from lv import zoo
from lv.ext.torchvision import models
from third_party.netdissect import renormalize

from torch.utils import data
from torchvision import datasets, transforms


def dissection_models() -> zoo.ModelConfigs:
    """Return configs for all models used in dissection."""
    return {
        'alexnet': {
            'imagenet':
                zoo.ModelConfig(models.alexnet_seq,
                                pretrained=True,
                                load_weights=False),
        },
        'resnet18': {
            'imagenet':
                zoo.ModelConfig(models.resnet18_seq,
                                pretrained=True,
                                load_weights=False)
        },
        'vgg16': {
            'imagenet':
                zoo.ModelConfig(models.vgg16_seq,
                                pretrained=True,
                                load_weights=False)
        }
    }


def dissection_datasets() -> zoo.DatasetConfigs:
    """Return configs for all datasets used in dissection."""
    # TODO(evandez): Are these the right transforms?
    return {
        'imagenet':
            zoo.DatasetConfig(datasets.ImageNet,
                              split='val',
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  renormalize.NORMALIZER['imagenet']
                              ])),
        'places365':
            zoo.DatasetConfig(datasets.Places365,
                              split='val',
                              download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor()
                              ])),
    }


def model(*args: Any, **kwargs: Any) -> zoo.Model:
    """Wrap `zoo.model` with a different default source."""
    kwargs.setdefault('source', dissection_models())
    return zoo.model(*args, **kwargs)


def dataset(*args: Any, **kwargs: Any) -> data.Dataset:
    """Wrap `zoo.dataset` with a different default source."""
    kwargs.setdefault('source', dissection_datasets())
    return zoo.dataset(*args, **kwargs)
