"""Model and dataset configs for computing exemplars."""
import dataclasses
from typing import Any, Mapping, Optional, Sequence, Tuple

from src import milannotations
from src.deps import alexnet, resnet152
from src.deps.ext.pretorched.gans import biggan
from src.deps.ext.torchvision import models
from src.deps.netdissect import renormalize
from src.exemplars import datasets, transforms
from src.utils import hubs
from src.utils.typing import Layer

import easydict
import torch
from torch import nn

HOST = f'{hubs.HOST}/exemplars/models'

KEYS = easydict.EasyDict(d=milannotations.KEYS)

LAYERS = easydict.EasyDict()
LAYERS.ALEXNET = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5')
LAYERS.BIGGAN = ('layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5')
LAYERS.DENSENET121 = (
    'features.conv0',
    *(f'features.denseblock{index}' for index in range(1, 5)))
LAYERS.DENSENET201 = LAYERS.DENSENET121
LAYERS.DINO_VITS8 = tuple(f'blocks.{layer}.mlp.fc1' for layer in range(12))
LAYERS.MOBILENET_V2 = (f'features.{index}' for index in range(0, 19, 2))
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
class ModelExemplarsConfig:
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


TransformInputs = transforms.TransformToTuple
TransformHiddens = transforms.TransformToTensor
TransformOutputs = transforms.TransformToTensor


@dataclasses.dataclass(frozen=True)
class DiscriminativeModelExemplarsConfig(ModelExemplarsConfig):
    """Dissection configuration for a discriminative model."""

    transform_inputs: Optional[TransformInputs] = None
    transform_hiddens: Optional[TransformHiddens] = None


@dataclasses.dataclass(frozen=True)
class GenerativeModelExemplarsConfig(ModelExemplarsConfig):
    """Dissection configuration for a model."""

    transform_inputs: Optional[TransformInputs] = None
    transform_hiddens: Optional[TransformHiddens] = None
    transform_outputs: Optional[TransformOutputs] = None

    # Special property: generative models want a dataset of representations,
    # not the dataset of images they were trained on. This is required.
    dataset: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the config."""
        if self.dataset is None:
            raise ValueError('GenerativeModelExemplarsConfig requires '
                             'dataset to be set')

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """Convert the config to kwargs."""
        kwargs = dict(super().kwargs)
        kwargs.pop('dataset', None)
        return kwargs


@dataclasses.dataclass
class ModelConfig(hubs.ModelConfig):
    """A model config that also stores dissection configuration."""

    def __init__(self,
                 *args: Any,
                 layers: Optional[Sequence[Layer]] = None,
                 exemplars: Optional[ModelExemplarsConfig] = None,
                 **kwargs: Any):
        """Initialize the config.

        Args:
            exemplars (Optional[Mapping[str, Any]]): Exemplars options.

        """
        super().__init__(*args, **kwargs)
        self.layers = layers
        self.exemplars = exemplars or ModelExemplarsConfig()


def model_hub() -> hubs.ModelHub:
    """Return configs for all models for which we can extract exemplars."""
    configs = {
        KEYS.ALEXNET_IMAGENET:
            ModelConfig(
                models.alexnet_seq,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.ALEXNET,
            ),
        KEYS.ALEXNET_IMAGENET_BLURRED:
            ModelConfig(
                models.alexnet,
                load_weights=True,
                layers=('features.0', 'features.3', 'features.6', 'features.8',
                        'features.10'),
            ),
        KEYS.ALEXNET_PLACES365:
            ModelConfig(
                alexnet.AlexNet,
                url=f'{HOST}/models/alexnet-places365.pth',
                transform_weights=lambda weights: weights['state_dict'],
                layers=LAYERS.ALEXNET),
        KEYS.BIGGAN_IMAGENET:
            ModelConfig(
                biggan.SeqBigGAN,
                pretrained='imagenet',
                load_weights=False,
                layers=LAYERS.BIGGAN,
                exemplars=GenerativeModelExemplarsConfig(
                    transform_inputs=lambda *xs: (biggan.GInputs(*xs),),
                    transform_hiddens=lambda hiddens: hiddens.h,
                    renormalizer=renormalize.renormalizer(target='byte'),
                    image_size=256,
                    batch_size=32,
                    dataset=datasets.KEYS.BIGGAN_ZS_IMAGENET,
                ),
            ),
        KEYS.BIGGAN_PLACES365:
            ModelConfig(
                biggan.SeqBigGAN,
                pretrained='places365',
                load_weights=False,
                layers=LAYERS.BIGGAN,
                exemplars=GenerativeModelExemplarsConfig(
                    transform_inputs=lambda *xs: (biggan.GInputs(*xs),),
                    transform_hiddens=lambda hiddens: hiddens.h,
                    renormalizer=renormalize.renormalizer(target='byte'),
                    image_size=256,
                    batch_size=32,
                    dataset=datasets.KEYS.BIGGAN_ZS_PLACES365,
                ),
            ),
        KEYS.DENSENET121_IMAGENET:
            ModelConfig(models.densenet121,
                        pretrained=True,
                        load_weights=False,
                        layers=LAYERS.DENSENET121),
        KEYS.DENSENET121_IMAGENET_BLURRED:
            ModelConfig(models.densenet121,
                        load_weights=True,
                        layers=LAYERS.DENSENET121),
        KEYS.DENSENET201_IMAGENET:
            ModelConfig(models.densenet201,
                        pretrained=True,
                        load_weights=False,
                        layers=LAYERS.DENSENET201),
        KEYS.DENSENET201_IMAGENET_BLURRED:
            ModelConfig(models.densenet201,
                        load_weights=True,
                        layers=LAYERS.DENSENET201),
        KEYS.DINO_VITS8_IMAGENET:
            ModelConfig(
                torch.hub.load,
                repo_or_dir='facebookresearch/dino:main',
                model=KEYS.DINO_VITS8,
                layers=LAYERS.DINO_VITS8,
                exemplars=DiscriminativeModelExemplarsConfig(
                    transform_hiddens=transforms.spatialize_vit_mlp,
                    batch_size=32),
                load_weights=False,
            ),
        KEYS.MOBILENET_V2_IMAGENET:
            ModelConfig(
                models.mobilenet_v2,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.MOBILENET_V2,
            ),
        KEYS.MOBILENET_V2_IMAGENET_BLURRED:
            ModelConfig(models.mobilenet_v2,
                        load_weights=True,
                        layers=LAYERS.MOBILENET_V2),
        KEYS.RESNET18_IMAGENET:
            ModelConfig(
                # TODO(evandez): No longer use seq version...
                models.resnet18_seq,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.RESNET18,
            ),
        KEYS.RESNET18_IMAGENET_BLURRED:
            ModelConfig(models.resnet18,
                        load_weights=True,
                        layers=LAYERS.RESNET18),
        KEYS.RESNET18_PLACES365:
            ModelConfig(
                models.resnet18,
                load_weights=True,
                layers=LAYERS.RESNET18,
                transform_weights=lambda weights: weights['state_dict'],
                url=f'{HOST}/models/resnet18-places365.pth',
                num_classes=365),
        KEYS.RESNET34_IMAGENET:
            ModelConfig(
                models.resnet34,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.RESNET34,
            ),
        KEYS.RESNET34_IMAGENET_BLURRED:
            ModelConfig(models.resnet34,
                        load_weights=True,
                        layers=LAYERS.RESNET34),
        KEYS.RESNET50_IMAGENET:
            ModelConfig(
                models.resnet50,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.RESNET50,
            ),
        KEYS.RESNET50_IMAGENET_BLURRED:
            ModelConfig(models.resnet50,
                        load_weights=True,
                        layers=LAYERS.RESNET50),
        KEYS.RESNET101_IMAGENET:
            ModelConfig(
                models.resnet101,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.RESNET101,
            ),
        KEYS.RESNET101_IMAGENET_BLURRED:
            ModelConfig(models.resnet101,
                        load_weights=True,
                        layers=LAYERS.RESNET101),
        KEYS.RESNET152_IMAGENET:
            ModelConfig(models.resnet152_seq,
                        pretrained=True,
                        load_weights=False,
                        layers=LAYERS.RESNET152),
        KEYS.RESNET152_IMAGENET_BLURRED:
            ModelConfig(models.resnet152,
                        load_weights=True,
                        layers=LAYERS.RESNET152),
        KEYS.RESNET152_PLACES365:
            ModelConfig(
                resnet152.OldResNet152,
                url=f'{HOST}/models/resnet152-places365.pth',
                layers=(0, 4, 5, 6, 7),
            ),
        KEYS.SHUFFLENET_V2_X1_0_IMAGENET:
            ModelConfig(
                models.shufflenet_v2_x1_0,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.SHUFFLENET_V2_X1_0,
            ),
        KEYS.SHUFFLENET_V2_X1_0_IMAGENET_BLURRED:
            ModelConfig(models.shufflenet_v2_x1_0,
                        load_weights=True,
                        layers=LAYERS.SHUFFLENET_V2_X1_0),
        KEYS.SQUEEZENET1_0_IMAGENET:
            ModelConfig(
                models.squeezenet1_0,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.SQUEEZENET1_0,
            ),
        KEYS.SQUEEZENET1_0_IMAGENET_BLURRED:
            ModelConfig(models.squeezenet1_0,
                        load_weights=True,
                        layers=LAYERS.SQUEEZENET1_0),
        KEYS.VGG11_IMAGENET:
            ModelConfig(
                models.vgg11,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.VGG11,
            ),
        KEYS.VGG11_IMAGENET_BLURRED:
            ModelConfig(
                models.vgg11,
                load_weights=True,
                layers=LAYERS.VGG11,
            ),
        KEYS.VGG13_IMAGENET:
            ModelConfig(
                models.vgg13,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.VGG13,
            ),
        KEYS.VGG13_IMAGENET_BLURRED:
            ModelConfig(
                models.vgg13,
                load_weights=True,
                layers=LAYERS.VGG13,
            ),
        KEYS.VGG16_IMAGENET:
            ModelConfig(
                models.vgg16,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.VGG16,
            ),
        KEYS.VGG16_IMAGENET_BLURRED:
            ModelConfig(
                models.vgg16,
                load_weights=True,
                layers=LAYERS.VGG16,
            ),
        KEYS.VGG19_IMAGENET:
            ModelConfig(
                models.vgg19,
                pretrained=True,
                load_weights=False,
                layers=LAYERS.VGG19,
            ),
        KEYS.VGG19_IMAGENET_BLURRED:
            ModelConfig(
                models.vgg19,
                load_weights=True,
                layers=LAYERS.VGG19,
            ),
    }
    return hubs.ModelHub(**configs)


Model = Tuple[nn.Module, Sequence[Layer], ModelConfig]


def load(name: str,
         hub: Optional[hubs.ModelHub] = None,
         **kwargs: Any) -> Model:
    """Load the model and also return its layers and config.

    Args:
        name (str): Model config name.
        hub (Optional[hubs.ModelHub], optional): Model hub to pull from.
            Defaults to one defined in this file.

    """
    if hub is None:
        hub = model_hub()
    model = hub.load(name, **kwargs)

    config = hub.configs[name]
    assert isinstance(config, ModelConfig), 'unknown config type'
    layers = config.layers
    if layers is None:
        layers = [key for key, _ in model.named_children()]

    return model, layers, config
