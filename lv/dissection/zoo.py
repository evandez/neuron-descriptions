"""Defines dissection configurations."""
import dataclasses
from typing import Any, Mapping, Optional, Sequence, Tuple

from lv import zoo
from lv.deps import alexnet, resnet152
from lv.deps.ext.pretorched.gans import biggan
from lv.deps.ext.torchvision import models
from lv.deps.netdissect import renormalize
from lv.dissection import datasets as lv_datasets
from lv.dissection import transforms as lv_transforms
from lv.utils.typing import Layer
from lv.zoo import (KEY_ALEXNET, KEY_BIGGAN, KEY_DINO_VITS8, KEY_IMAGENET,
                    KEY_PLACES365, KEY_RESNET152)

import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms

LV_HOST = 'https://unitname.csail.mit.edu/dissect/models'
DISSECT_HOST = 'https://dissect.csail.mit.edu/models'

KEY_RESNET18 = 'resnet18'

KEY_SPURIOUS_IMAGENET_TEXT = 'spurious-imagenet-text'

KEY_RESNET18_IMAGENET = f'{KEY_RESNET18}/{KEY_IMAGENET}'
KEY_RESNET18_PLACES365 = f'{KEY_RESNET18}/{KEY_PLACES365}'
KEY_BIGGAN_ZS_IMAGENET = 'biggan-zs-imagenet'
KEY_BIGGAN_ZS_PLACES365 = 'biggan-zs-places365'

LAYERS_ALEXNET = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5')
LAYERS_RESNET18 = ('conv1', 'layer1', 'layer2', 'layer3', 'layer4')
LAYERS_RESNET152 = LAYERS_RESNET18
LAYERS_BIGGAN = ('layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5')
LAYERS_DINO_VITS8 = tuple(f'blocks.{layer}.mlp.fc1' for layer in range(12))


@dataclasses.dataclass(frozen=True)
class ModelDissectionConfig:
    """Generic dissection configuration."""

    k: Optional[int] = None
    quantile: Optional[float] = None
    output_size: Optional[int] = None
    batch_size: Optional[int] = None
    image_size: Optional[int] = None
    renormalizer: Optional[renormalize.Renormalizer] = None

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Convert the config to kwargs."""
        kwargs = {}
        for key, value in vars(self).items():
            if value is not None:
                kwargs[key] = value
        return kwargs


@dataclasses.dataclass(frozen=True)
class DiscriminativeModelDissectionConfig(ModelDissectionConfig):
    """Dissection configuration for a discriminative model."""

    transform_inputs: Optional[lv_transforms.TransformToTuple] = None
    transform_hiddens: Optional[lv_transforms.TransformToTensor] = None


@dataclasses.dataclass(frozen=True)
class GenerativeModelDissectionConfig(ModelDissectionConfig):
    """Dissection configuration for a model."""

    transform_inputs: Optional[lv_transforms.TransformToTuple] = None
    transform_hiddens: Optional[lv_transforms.TransformToTensor] = None
    transform_outputs: Optional[lv_transforms.TransformToTensor] = None

    # Special property: generative models want a dataset of representations,
    # not the dataset of images they were trained on. This is required.
    dataset: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the config."""
        if self.dataset is None:
            raise ValueError('GenerativeModelDissectionConfig requires '
                             'dataset to be set')

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Convert the config to kwargs."""
        kwargs = dict(super().kwargs)
        kwargs.pop('dataset', None)
        return kwargs


@dataclasses.dataclass
class ModelConfig(zoo.ModelConfig):
    """A model config that also stores dissection configuration."""

    def __init__(self,
                 *args: Any,
                 layers: Optional[Sequence[Layer]] = None,
                 dissection: Optional[ModelDissectionConfig] = None,
                 **kwargs: Any):
        """Initialize the config.

        Args:
            dissection (Optional[Mapping[str, Any]]): Dissection options.

        """
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.dissection = dissection or ModelDissectionConfig()


Model = Tuple[nn.Module, Sequence[Layer], ModelConfig]
ModelConfigs = Mapping[str, Mapping[str, ModelConfig]]


def dissection_models() -> ModelConfigs:
    """Return configs for all models used in dissection."""
    return {
        KEY_ALEXNET: {
            KEY_IMAGENET:
                ModelConfig(models.alexnet_seq,
                            pretrained=True,
                            load_weights=False,
                            layers=LAYERS_ALEXNET),
            KEY_PLACES365:
                ModelConfig(
                    alexnet.AlexNet,
                    url=f'{LV_HOST}/alexnet-places365.pth',
                    transform_weights=lambda weights: weights['state_dict'],
                    layers=LAYERS_ALEXNET),
        },
        KEY_RESNET18: {
            KEY_IMAGENET:
                ModelConfig(
                    models.resnet18_seq,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS_RESNET18,
                    dissection=DiscriminativeModelDissectionConfig(
                        image_size=224,
                        renormalizer=renormalize.renormalizer(
                            source='imagenet', target='byte'),
                    ),
                ),
        },
        KEY_RESNET152: {
            KEY_IMAGENET:
                ModelConfig(models.resnet152_seq,
                            pretrained=True,
                            load_weights=False,
                            layers=LAYERS_RESNET152),
            KEY_PLACES365:
                ModelConfig(
                    resnet152.OldResNet152,
                    url=f'{DISSECT_HOST}/resnet152_places365-f928166e5c.pth',
                    layers=(0, 4, 5, 6, 7),
                ),
        },
        KEY_BIGGAN: {
            KEY_IMAGENET:
                ModelConfig(
                    biggan.SeqBigGAN,
                    pretrained='imagenet',
                    load_weights=False,
                    layers=LAYERS_BIGGAN,
                    dissection=GenerativeModelDissectionConfig(
                        transform_inputs=lambda *xs: (biggan.GInputs(*xs),),
                        transform_hiddens=lambda hiddens: hiddens.h,
                        renormalizer=renormalize.renormalizer(target='byte'),
                        image_size=256,
                        batch_size=32,
                        dataset=KEY_BIGGAN_ZS_IMAGENET,
                    ),
                ),
            KEY_PLACES365:
                ModelConfig(
                    biggan.SeqBigGAN,
                    pretrained='imagenet',
                    load_weights=False,
                    layers=LAYERS_BIGGAN,
                    dissection=GenerativeModelDissectionConfig(
                        transform_inputs=lambda *xs: (biggan.GInputs(*xs),),
                        transform_hiddens=lambda hiddens: hiddens.h,
                        renormalizer=renormalize.renormalizer(target='byte'),
                        image_size=256,
                        batch_size=32,
                        dataset=KEY_BIGGAN_ZS_IMAGENET,
                    ),
                ),
        },
        KEY_DINO_VITS8: {
            KEY_IMAGENET:
                ModelConfig(
                    torch.hub.load,
                    repo_or_dir='facebookresearch/dino:main',
                    model=KEY_DINO_VITS8,
                    layers=LAYERS_DINO_VITS8,
                    dissection=DiscriminativeModelDissectionConfig(
                        transform_hiddens=lv_transforms.spatialize_vit_mlp,
                        batch_size=32),
                    load_weights=False,
                ),
        },
    }


def dissection_datasets() -> zoo.DatasetConfigs:
    """Return configs for all datasets used in dissection."""
    return {
        zoo.KEY_IMAGENET:
            zoo.DatasetConfig(datasets.ImageFolder,
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  renormalize.NORMALIZER['imagenet']
                              ])),
        zoo.KEY_PLACES365:
            zoo.DatasetConfig(datasets.ImageFolder,
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  renormalize.NORMALIZER['imagenet'],
                              ])),
        KEY_SPURIOUS_IMAGENET_TEXT:
            zoo.DatasetConfig(datasets.ImageFolder,
                              transform=transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  renormalize.NORMALIZER['imagenet']
                              ])),
        KEY_BIGGAN_ZS_IMAGENET:
            zoo.DatasetConfig(lv_datasets.TensorDatasetOnDisk,
                              url=f'{LV_HOST}/{KEY_BIGGAN_ZS_IMAGENET}.pth'),
        KEY_BIGGAN_ZS_PLACES365:
            zoo.DatasetConfig(lv_datasets.TensorDatasetOnDisk,
                              url=f'{LV_HOST}/{KEY_BIGGAN_ZS_PLACES365}.pth'),
    }


def model(*args: Any, **kwargs: Any) -> Model:
    """Wrap `zoo.model` with a different default source."""
    kwargs.setdefault('source', dissection_models())
    model, config = zoo.model(*args, **kwargs)
    assert isinstance(config, ModelConfig), 'unknown config type'
    layers = config.layers
    if layers is None:
        layers = [key for key, _ in model.named_children()]
    return model, layers, config


def dataset(*args: Any, **kwargs: Any) -> data.Dataset:
    """Wrap `zoo.dataset` with a different default source."""
    kwargs.setdefault('source', dissection_datasets())
    return zoo.dataset(*args, **kwargs)
