"""Defines dissection configurations."""
from typing import Any

from lv import zoo
from lv.ext.torchvision import models
from third_party import alexnet
from third_party.netdissect import renormalize

from torch.utils import data
from torchvision import datasets, transforms

MODEL_HOST = 'http://wednesday.csail.mit.edu/dez/vocabulary/models'
MODEL_FILE_PLACES = 'iter_131072_weights.pth'

KEY_ALEXNET = 'alexnet'
KEY_RESNET18 = 'resnet18'
KEY_VGG_16 = 'vgg16'

KEY_IMAGENET = 'imagenet'
KEY_PLACES365 = 'places365'

CONV_LAYERS_ALEXNET = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5')
CONV_LAYER_RESNET18 = ('conv1', 'layer1', 'layer2', 'layer3', 'layer4')
CONV_LAYERS_VGG16 = ('conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                     'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
                     'conv5_1', 'conv5_2', 'conv5_3')


def dissection_models() -> zoo.ModelConfigs:
    """Return configs for all models used in dissection."""
    return {
        KEY_ALEXNET: {
            KEY_IMAGENET:
                zoo.ModelConfig(models.alexnet_seq,
                                pretrained=True,
                                load_weights=False,
                                layers=CONV_LAYERS_ALEXNET),
            KEY_PLACES365:
                zoo.ModelConfig(
                    alexnet.AlexNet,
                    url=f'{MODEL_HOST}/alexnet/places/{MODEL_FILE_PLACES}',
                    transform_weights=lambda weights: weights['state_dict'],
                    layers=CONV_LAYERS_ALEXNET)
        },
        KEY_RESNET18: {
            KEY_IMAGENET:
                zoo.ModelConfig(models.resnet18_seq,
                                pretrained=True,
                                load_weights=False,
                                layers=CONV_LAYER_RESNET18),
            KEY_PLACES365:
                zoo.ModelConfig(
                    models.resnet18_seq,
                    num_classes=365,
                    url=f'{MODEL_HOST}/resnet18/places/{MODEL_FILE_PLACES}',
                    transform_weights=lambda weights: weights['state_dict'],
                    layers=CONV_LAYER_RESNET18),
        },
        KEY_VGG_16: {
            KEY_IMAGENET:
                zoo.ModelConfig(models.vgg16_seq,
                                pretrained=True,
                                load_weights=False,
                                layers=CONV_LAYERS_VGG16),
            KEY_PLACES365:
                zoo.ModelConfig(
                    models.vgg16_seq,
                    num_classes=365,
                    url=f'{MODEL_HOST}/vgg16/places/{MODEL_FILE_PLACES}',
                    transform_weights=lambda weights: weights['state_dict'],
                    layers=CONV_LAYERS_VGG16),
        },
    }


def dissection_datasets() -> zoo.DatasetConfigs:
    """Return configs for all datasets used in dissection."""
    # TODO(evandez): Are these the right transforms?
    return {
        KEY_IMAGENET:
            zoo.DatasetConfig(datasets.ImageFolder,
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  renormalize.NORMALIZER['imagenet']
                              ])),
        # TODO(evandez): This uses ImageFolder to be backwards compatible,
        # but we should probably use datasets.Places365 in the final version.
        KEY_PLACES365:
            zoo.DatasetConfig(datasets.ImageFolder,
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
