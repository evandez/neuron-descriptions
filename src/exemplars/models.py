"""Configs describing how to compute exemplars for each model.

The most important function here is `model_hub`, which returns a mapping
from model name (formatted as <model architecture>/<training dataset>)
to a special config object. The config object is described more in
`src/utils/hubs.py`, but the most important thing to know is it takes
an arbitrary factory function for the model and, optionally, will look
for pretrained weights at $MILAN_MODELS_DIR/model_name.pth if
load_weights=True (though this path can be overwritten at runtime).

Additionally, the configs allow you to specify the *layers* to compute
exemplars for (by default, all of them). These must be a fully specified
path to the torch submodule, as they will be read using PyTorch hooks. To
see the full list of possible layers for your model, look at
`your_model.named_parameters()`.
"""
import dataclasses
from typing import Any, Mapping, Optional, Sequence, Tuple

from src import milannotations
from src.deps import alexnet, resnet152
from src.deps.ext.pretorched.gans import biggan
from src.deps.ext.torchvision import models
from src.deps.netdissect import renormalize
from src.exemplars import datasets, transforms
from src.utils import hubs
from src.utils.typing import Layer, StateDict

import easydict
import torch
from torch import nn

# We don't host most of these models, either the NetDissect team does
# or the torchvision people.
HOST = 'https://dissect.csail.mit.edu/models'

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


def rekey_vgg16(state_dict: StateDict) -> StateDict:
    """Convert places365-style vgg16 state dict to torchvision-style."""
    mappings = dict([('conv1_1', '0'), ('conv1_2', '2'), ('conv2_1', '5'),
                     ('conv2_2', '7'), ('conv3_1', '10'), ('conv3_2', '12'),
                     ('conv3_3', '14'), ('conv4_1', '17'), ('conv4_2', '19'),
                     ('conv4_3', '21'), ('conv5_1', '24'), ('conv5_2', '26'),
                     ('conv5_3', '28'), ('fc6', '0'), ('fc7', '3'),
                     ('fc8', '6'), ('fc8a', '6')])

    def translate_name(name: str) -> str:
        parts = name.split('.')
        if parts[1] in mappings:
            parts[1] = mappings[parts[1]]
        return '.'.join(parts)

    return {translate_name(k): v for k, v in state_dict.items()}


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


def default_model_configs(**others: ModelConfig) -> Mapping[str, ModelConfig]:
    """Return the default model configs."""
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
                url=f'{hubs.HOST}/models/alexnet-places365.pth',
                transform_weights=lambda weights: weights['state_dict'],
                layers=LAYERS.ALEXNET),
        KEYS.VGG16_PLACES365:
            ModelConfig(models.vgg16,
                        url=f'{HOST}/vgg16_places365-0bafbc55.pth',
                        transform_weights=rekey_vgg16,
                        layers=LAYERS.VGG16,
                        num_classes=365),
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
                url=f'{HOST}/resnet18_places365-2f475921.pth',
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
                url=f'{HOST}/resnet152_places365-f928166e5c.pth',
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
    configs.update(others)
    return configs


def default_model_hub(**others: ModelConfig) -> hubs.ModelHub:
    """Return configs for all models for which we can extract exemplars."""
    configs = default_model_configs(**others)
    return hubs.ModelHub(**configs)


Model = Tuple[nn.Module, Sequence[Layer], ModelConfig]


def load(name: str,
         configs: Optional[Mapping[str, ModelConfig]] = None,
         **kwargs: Any) -> Model:
    """Load the model and also return its layers and config.

    Args:
        name (str): Model config name.
        configs (Optional[Mapping[str, ModelConfig]], optional): Model configs
            to use when loading models, in addition to those returned by
            default_model_hub(). Defaults to just those returned by
            default_model_hub().

    Returns:
        Model: The loaded model, it's default exemplar-able layers, and its
            config.

    """
    configs = configs or {}
    hub = default_model_hub(**configs)
    model = hub.load(name, **kwargs)

    config = hub.configs[name]
    assert isinstance(config, ModelConfig), 'unknown config type'
    layers = config.layers
    if layers is None:
        layers = [key for key, _ in model.named_children()]

    return model, layers, config
