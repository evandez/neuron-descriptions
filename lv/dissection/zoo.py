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

import easydict
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms

LV_HOST = 'https://unitname.csail.mit.edu/dissect/models'
DISSECT_HOST = 'https://dissect.csail.mit.edu/models'

KEYS = easydict.EasyDict(d=zoo.KEYS)
KEYS.IMAGENET_SPURIOUS_TEXT = 'imagenet-spurious-text'
KEYS.IMAGENET_SPURIOUS_COLOR = 'imagenet-spurious-color'
KEYS.BIGGAN_ZS_IMAGENET = 'biggan-zs-imagenet'
KEYS.BIGGAN_ZS_PLACES365 = 'biggan-zs-places365'

LAYERS = easydict.EasyDict()
LAYERS.ALEXNET = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5')
LAYERS.BIGGAN = ('layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5')
LAYERS.DENSENET121 = (
    'features.conv0',
    *(f'features.denseblock{index}' for index in range(1, 5)))
LAYERS.DENSENET201 = LAYERS.DENSENET121
LAYERS.DINO_VITS8 = tuple(f'blocks.{layer}.mlp.fc1' for layer in range(12))
LAYERS.MOBILENET_V2 = (f'features.{index}' for index in range(18))
LAYERS.RESNET18 = ('conv1', 'layer1', 'layer2', 'layer3', 'layer4')
LAYERS.RESNET34 = LAYERS.RESNET18
LAYERS.RESNET50 = LAYERS.RESNET18
LAYERS.RESNET101 = LAYERS.RESNET18
LAYERS.RESNET152 = LAYERS.RESNET18
LAYERS.SHUFFLENET_V2_X1_0 = ('conv1', 'stage2', 'stage3', 'stage4', 'conv5')
LAYERS.SQUEEZENET1_0 = (
    f'features.{index}' for index in (0, 3, 4, 5, 7, 8, 9, 10, 12))
LAYERS.VGG11 = tuple(f'features.{index}' for index in (0, 3, 8, 13, 18))
LAYERS.VGG13 = tuple(f'features.{index}' for index in (2, 7, 12, 17, 22))
LAYERS.VGG16 = tuple(f'features.{index}' for index in (2, 7, 14, 21, 28))
LAYERS.VGG19 = tuple(f'features.{index}' for index in (2, 7, 16, 25, 34))


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
        KEYS.ALEXNET: {
            KEYS.IMAGENET:
                ModelConfig(models.alexnet_seq,
                            pretrained=True,
                            load_weights=False,
                            layers=LAYERS.ALEXNET),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.alexnet,
                            load_weights=True,
                            layers=('features.0', 'features.3', 'features.6',
                                    'features.8', 'features.10')),
            KEYS.PLACES365:
                ModelConfig(
                    alexnet.AlexNet,
                    url=f'{LV_HOST}/alexnet-places365.pth',
                    transform_weights=lambda weights: weights['state_dict'],
                    layers=LAYERS.ALEXNET),
        },
        KEYS.BIGGAN: {
            KEYS.IMAGENET:
                ModelConfig(
                    biggan.SeqBigGAN,
                    pretrained='imagenet',
                    load_weights=False,
                    layers=LAYERS.BIGGAN,
                    dissection=GenerativeModelDissectionConfig(
                        transform_inputs=lambda *xs: (biggan.GInputs(*xs),),
                        transform_hiddens=lambda hiddens: hiddens.h,
                        renormalizer=renormalize.renormalizer(target='byte'),
                        image_size=256,
                        batch_size=32,
                        dataset=KEYS.BIGGAN_ZS_IMAGENET,
                    ),
                ),
            KEYS.PLACES365:
                ModelConfig(
                    biggan.SeqBigGAN,
                    pretrained='imagenet',
                    load_weights=False,
                    layers=LAYERS.BIGGAN,
                    dissection=GenerativeModelDissectionConfig(
                        transform_inputs=lambda *xs: (biggan.GInputs(*xs),),
                        transform_hiddens=lambda hiddens: hiddens.h,
                        renormalizer=renormalize.renormalizer(target='byte'),
                        image_size=256,
                        batch_size=32,
                        dataset=KEYS.BIGGAN_ZS_IMAGENET,
                    ),
                ),
        },
        KEYS.DENSENET121: {
            KEYS.IMAGENET:
                ModelConfig(models.densenet121,
                            pretrained=True,
                            load_weights=False,
                            layers=LAYERS.DENSENET121),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.densenet121,
                            load_weights=True,
                            layers=LAYERS.DENSENET121),
        },
        KEYS.DENSENET201: {
            KEYS.IMAGENET:
                ModelConfig(models.densenet201,
                            pretrained=True,
                            load_weights=False,
                            layers=LAYERS.DENSENET201),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.densenet201,
                            load_weights=True,
                            layers=LAYERS.DENSENET201),
        },
        KEYS.DINO_VITS8: {
            KEYS.IMAGENET:
                ModelConfig(
                    torch.hub.load,
                    repo_or_dir='facebookresearch/dino:main',
                    model=KEYS.DINO_VITS8,
                    layers=LAYERS.DINO_VITS8,
                    dissection=DiscriminativeModelDissectionConfig(
                        transform_hiddens=lv_transforms.spatialize_vit_mlp,
                        batch_size=32),
                    load_weights=False,
                ),
        },
        KEYS.MOBILENET_V2: {
            KEYS.IMAGENET:
                ModelConfig(
                    models.mobilenet_v2,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS.MOBILENET_V2,
                ),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.mobilenet_v2,
                            load_weights=True,
                            layers=LAYERS.MOBILENET_V2),
        },
        KEYS.RESNET18: {
            KEYS.IMAGENET:
                ModelConfig(
                    # TODO(evandez): No longer use seq version...
                    models.resnet18_seq,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS.RESNET18,
                ),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.resnet18,
                            load_weights=True,
                            layers=LAYERS.RESNET18),
        },
        KEYS.RESNET34: {
            KEYS.IMAGENET:
                ModelConfig(
                    models.resnet34,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS.RESNET34,
                ),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.resnet34,
                            load_weights=True,
                            layers=LAYERS.RESNET34),
        },
        KEYS.RESNET50: {
            KEYS.IMAGENET:
                ModelConfig(
                    models.resnet50,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS.RESNET50,
                ),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.resnet50,
                            load_weights=True,
                            layers=LAYERS.RESNET50),
        },
        KEYS.RESNET101: {
            KEYS.IMAGENET:
                ModelConfig(
                    models.resnet101,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS.RESNET101,
                ),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.resnet101,
                            load_weights=True,
                            layers=LAYERS.RESNET101),
        },
        KEYS.RESNET152: {
            KEYS.IMAGENET:
                ModelConfig(models.resnet152_seq,
                            pretrained=True,
                            load_weights=False,
                            layers=LAYERS.RESNET152),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.resnet152,
                            load_weights=True,
                            layers=LAYERS.RESNET152),
            KEYS.PLACES365:
                ModelConfig(
                    resnet152.OldResNet152,
                    url=f'{DISSECT_HOST}/resnet152_places365-f928166e5c.pth',
                    layers=(0, 4, 5, 6, 7),
                ),
        },
        KEYS.SHUFFLENET_V2_X1_0: {
            KEYS.IMAGENET:
                ModelConfig(
                    models.shufflenet_v2_x1_0,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS.SHUFFLENET_V2_X1_0,
                ),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.shufflenet_v2_x1_0,
                            load_weights=True,
                            layers=LAYERS.SHUFFLENET_V2_X1_0),
        },
        KEYS.SQUEEZENET1_0: {
            KEYS.IMAGENET:
                ModelConfig(
                    models.squeezenet1_0,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS.SQUEEZENET1_0,
                ),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.squeezenet1_0,
                            load_weights=True,
                            layers=LAYERS.SQUEEZENET1_0),
        },
        KEYS.VGG11: {
            KEYS.IMAGENET:
                ModelConfig(
                    models.vgg11,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS.VGG11,
                ),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.vgg11,
                            load_weights=True,
                            layers=LAYERS.VGG11),
        },
        KEYS.VGG13: {
            KEYS.IMAGENET:
                ModelConfig(
                    models.vgg13,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS.VGG13,
                ),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.vgg13,
                            load_weights=True,
                            layers=LAYERS.VGG13),
        },
        KEYS.VGG16: {
            KEYS.IMAGENET:
                ModelConfig(
                    models.vgg16,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS.VGG16,
                ),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.vgg16,
                            load_weights=True,
                            layers=LAYERS.VGG16),
        },
        KEYS.VGG19: {
            KEYS.IMAGENET:
                ModelConfig(
                    models.vgg19,
                    pretrained=True,
                    load_weights=False,
                    layers=LAYERS.VGG19,
                ),
            KEYS.IMAGENET_BLURRED:
                ModelConfig(models.vgg19,
                            load_weights=True,
                            layers=LAYERS.VGG19),
        },
    }


def dissection_datasets() -> zoo.DatasetConfigs:
    """Return configs for all datasets used in dissection."""
    return {
        zoo.KEYS.IMAGENET:
            zoo.DatasetConfig(datasets.ImageFolder,
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  renormalize.NORMALIZER['imagenet']
                              ])),
        zoo.KEYS.PLACES365:
            zoo.DatasetConfig(datasets.ImageFolder,
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  renormalize.NORMALIZER['imagenet'],
                              ])),
        KEYS.IMAGENET_SPURIOUS_TEXT:
            zoo.DatasetConfig(datasets.ImageFolder,
                              transform=transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  renormalize.NORMALIZER['imagenet']
                              ])),
        KEYS.IMAGENET_SPURIOUS_COLOR:
            zoo.DatasetConfig(datasets.ImageFolder,
                              transform=transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  renormalize.NORMALIZER['imagenet']
                              ])),
        KEYS.BIGGAN_ZS_IMAGENET:
            zoo.DatasetConfig(lv_datasets.TensorDatasetOnDisk,
                              url=f'{LV_HOST}/{KEYS.BIGGAN_ZS_IMAGENET}.pth'),
        KEYS.BIGGAN_ZS_PLACES365:
            zoo.DatasetConfig(lv_datasets.TensorDatasetOnDisk,
                              url=f'{LV_HOST}/{KEYS.BIGGAN_ZS_PLACES365}.pth'),
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
