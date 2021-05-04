"""Defines dissection configurations."""
import dataclasses
from typing import Any, Iterable, Mapping, Optional, Tuple

from lv import zoo
from lv.dissection import datasets as lv_datasets
from lv.dissection import transforms as lv_transforms
from lv.ext import resnet152
from lv.ext.pretorched.gans import biggan
from lv.ext.torchvision import models
from lv.utils.typing import Layer
from third_party import alexnet
from third_party.netdissect import renormalize

from torch import nn
from torch.utils import data
from torchvision import datasets, transforms

# TODO(evandez): Migrate to the host below.
# LV_HOST = 'https://unitname.csail.mit.edu/dissect/models'
LV_HOST = 'http://wednesday.csail.mit.edu/dez/latent-vocabulary/dissect/models'
DISSECT_HOST = 'https://dissect.csail.mit.edu/models'

KEY_ALEXNET = 'alexnet'
KEY_RESNET18 = 'resnet18'
KEY_RESNET152 = 'resnet152'
KEY_VGG_16 = 'vgg16'
KEY_BIGGAN = 'biggan'

KEY_IMAGENET = 'imagenet'
KEY_PLACES365 = 'places365'
KEY_BIGGAN_ZS_IMAGENET = 'biggan-zs-imagenet'
KEY_BIGGAN_ZS_PLACES365 = 'biggan-zs-places365'

LAYERS_ALEXNET = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5')
LAYERS_RESNET18 = ('conv1', 'layer1', 'layer2', 'layer3', 'layer4')
LAYERS_RESNET152 = LAYERS_RESNET18
LAYERS_VGG16 = ('conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
                'conv5_1', 'conv5_2', 'conv5_3')
LAYERS_BIGGAN = ('layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5')


@dataclasses.dataclass(frozen=True)
class ModelDissectionConfig:
    """Dissection configuration for a model."""

    generative: bool = False
    transform_inputs: Optional[lv_transforms.TransformToTuple] = None
    transform_hiddens: Optional[lv_transforms.TransformToTensor] = None
    transform_outputs: Optional[lv_transforms.TransformToTensor] = None

    def __post_init__(self) -> None:
        """Validate the config."""
        if not self.generative and self.transform_hiddens is not None:
            raise ValueError('can only set transform_hiddens if generative')

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Convert the config to kwargs."""
        kwargs = {}
        for key, value in vars(self).items():
            if key.startswith('transform') and value is not None:
                kwargs[key] = value
        return kwargs


@dataclasses.dataclass
class ModelConfig(zoo.ModelConfig):
    """A model config that also stores dissection configuration."""

    def __init__(self,
                 *args: Any,
                 dissection: Optional[ModelDissectionConfig] = None,
                 **kwargs: Any):
        """Initialize the config.

        Args:
            dissection (Optional[Mapping[str, Any]]): Dissection options.

        """
        super().__init__(*args, **kwargs)
        self.dissection = dissection or ModelDissectionConfig()


Model = Tuple[nn.Sequential, Iterable[Layer], ModelConfig]
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
                    layers=LAYERS_ALEXNET)
        },
        KEY_RESNET18: {
            KEY_IMAGENET:
                ModelConfig(models.resnet18_seq,
                            pretrained=True,
                            load_weights=False,
                            layers=LAYERS_RESNET18),
            KEY_PLACES365:
                ModelConfig(
                    models.resnet18_seq,
                    num_classes=365,
                    url=f'{LV_HOST}/resnet18-places365.pth',
                    transform_weights=lambda weights: weights['state_dict'],
                    layers=LAYERS_RESNET18),
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
                    layers=LAYERS_RESNET152,
                ),
        },
        KEY_VGG_16: {
            KEY_IMAGENET:
                ModelConfig(models.vgg16_seq,
                            pretrained=True,
                            load_weights=False,
                            layers=LAYERS_VGG16),
        },
        KEY_BIGGAN: {
            KEY_IMAGENET:
                ModelConfig(
                    biggan.SeqBigGAN,
                    pretrained='imagenet',
                    load_weights=False,
                    layers=LAYERS_BIGGAN,
                    dissection=ModelDissectionConfig(
                        generative=True,
                        transform_inputs=lambda *xs: (biggan.GInputs(*xs),),
                        transform_hiddens=lambda hiddens: hiddens.h,
                    ),
                ),
            KEY_PLACES365:
                ModelConfig(
                    biggan.SeqBigGAN,
                    pretrained='places365',
                    load_weights=False,
                    layers=LAYERS_BIGGAN,
                    dissection=ModelDissectionConfig(
                        generative=True,
                        transform_inputs=lambda *xs: (biggan.GInputs(*xs),),
                        transform_hiddens=lambda hiddens: hiddens.h,
                    ),
                ),
        }
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
    model, layers, config = zoo.model(*args, **kwargs)
    assert isinstance(config, ModelConfig), 'unknown config type'
    return model, layers, config


def dataset(*args: Any, **kwargs: Any) -> data.Dataset:
    """Wrap `zoo.dataset` with a different default source."""
    kwargs.setdefault('source', dissection_datasets())
    return zoo.dataset(*args, **kwargs)
