"""The visual encoder component of MILAN.

These map image regions (given by an image and a mask) to a single vector.
They do so by feeding the images to a pretrained image classifier, reading
its intermediate features, and applying the mask to those features.
"""
from typing import (Any, Callable, Mapping, Optional, Sequence, Tuple, Type,
                    Union, overload)

from src.deps.netdissect import nethook, renormalize
from src.milannotations import datasets
from src.utils import serialize
from src.utils.typing import Device

import torch
from torch import nn
from torch.nn import functional
from torch.utils import data
from torchvision import models
from tqdm.auto import tqdm


class Encoder(serialize.SerializableModule):
    """An abstract module mapping images (and optionally masks) to features."""

    feature_shape: Tuple[int, ...]

    @overload
    def forward(self, images: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Featurize the given images.

        Equivalent to featurizing the images where the masks are all 1s.

        Args:
            images (torch.Tensor): The images to encoder. Should have
                shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: The image features. Will have shape
                (batch_size, *feature_shape).

        """
        ...

    @overload
    def forward(self, images: torch.Tensor, masks: torch.Tensor,
                **kwargs: Any) -> torch.Tensor:
        ...

    def forward(self,
                images: torch.Tensor,
                masks: Optional[torch.Tensor] = None,
                **kwargs: Any) -> torch.Tensor:
        """Abstract forward function corresponding to both overloads."""
        raise NotImplementedError

    def properties(self) -> serialize.Properties:
        """Require subclasses to implement `Serializable.properties`."""
        raise NotImplementedError

    def map(self,
            dataset: data.Dataset,
            mask: bool = True,
            image_index: Union[int, str] = -3,
            mask_index: Union[int, str] = -2,
            batch_size: int = 64,
            num_workers: int = 0,
            device: Optional[Device] = None,
            display_progress_as: Union[bool, str] = True,
            **kwargs: Any) -> data.TensorDataset:
        """Featurize an entire dataset.

        Keyword arguments are passed to `forward`.

        Args:
            dataset (data.Dataset): The dataset to featurize. Should return
                a sequence or mapping of values.
            mask (bool, optional): Try to read masks from batch and pass them
                to the encoder. Setting this to False is equivalent to
                setting masks to be all 1s. Defaults to True.
            image_index (int, optional): Index of image in each dataset
                sample. Defaults to -3 to be compatible with
                AnnotatedTopImagesDataset.
            mask_index (int, optional): Index of mask in each dataset
                sample. Defaults to -2 to be compatible with
                AnnotatedTopImagesDataset.
            batch_size (int, optional): Featurize images in batches of this
                size. Defaults to 64.
            num_workers (int, optional): Number of workers for loading data.
                Defaults to 0.
            device (Optional[Device], optional): Run preprocessing on this
                device. Defaults to None.
            display_progress_as (Union[bool, str], optional): If a string, show
                progress bar with this key. If True, show progress bar and
                generate the key. If False, do not show progress bar. Defaults
                to True.

        Raises:
            ValueError: If images or masks are not tensors.

        Returns:
            data.TensorDataset: Dataset of image features.

        """
        if device is not None:
            self.to(device)

        mapped = []

        loader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
        if isinstance(display_progress_as, str) or display_progress_as:
            if not isinstance(display_progress_as, str):
                if isinstance(dataset, (datasets.TopImagesDataset,
                                        datasets.AnnotatedTopImagesDataset)):
                    display_progress_as = f'featurize {dataset.name}'
                else:
                    display_progress_as = 'featurize dataset'
            loader = tqdm(loader, desc=display_progress_as)

        for batch in loader:
            images = batch[image_index]
            if not isinstance(images, torch.Tensor):
                raise ValueError(f'non-tensor images: {type(images).__name__}')
            if device is not None:
                images = images.to(device)
            inputs = [images.view(-1, *images.shape[-3:])]

            masks = None
            if mask:
                masks = batch[mask_index]
                if not isinstance(masks, torch.Tensor):
                    raise ValueError(
                        f'non-tensor masks: {type(masks).__name__}')
                if device is not None:
                    masks = masks.to(device)
                inputs.append(masks.view(-1, *masks.shape[-3:]))

            with torch.no_grad():
                features = self(*inputs, **kwargs)

            # Unflatten the outputs.
            features = features.view(*images.shape[:-3], *self.feature_shape)

            mapped.append(features)

        return data.TensorDataset(torch.cat(mapped))


ClassifierFactory = Callable[..., nn.Sequential]
ClassifierLayers = Sequence[str]
ClassifierNumFeatures = int
ClassifierFeatureSize = int
SpatialConvEncoderConfig = Tuple[ClassifierFactory, ClassifierLayers,
                                 ClassifierNumFeatures, ClassifierFeatureSize]


class SpatialConvEncoder(Encoder):
    """Encodes images spatially using conv features from a pretrained CNN."""

    def __init__(self, config: str = 'resnet18', **kwargs: Any):
        """Initialize the encoder.

        Keyword arguments are forwarded to the constructor of the underlying
        classifier model.

        Args:
            config (str, optional): The encoder config to use.
                See `SpatialConvEncoder.configs` for options.
                Defaults to 'resnet18'.

        """
        super().__init__()

        configs = SpatialConvEncoder.configs()
        if config not in configs:
            raise ValueError(f'encoder not supported: {config}')

        self.config = config
        self.kwargs = kwargs
        self.kwargs.setdefault('pretrained', True)

        factory, layers, n_features, feature_size = configs[config]
        assert len(layers) == 1, 'multiple layers?'
        layer, = layers

        self.encoder = nethook.InstrumentedModel(factory(**self.kwargs))
        self.encoder.retain_layer(layer)
        self.encoder.eval()

        self.layer = layer
        self.feature_shape = (n_features, feature_size)

        # We will manually normalize input images. This just makes life easier.
        mean, std = renormalize.OFFSET_SCALE['imagenet']
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self,
                images: torch.Tensor,
                masks: Optional[torch.Tensor] = None,
                normalize: bool = True,
                **_: Any) -> torch.Tensor:
        """Encode the images."""
        if masks is None:
            masks = images.new_ones((len(images), 1, *images.shape[2:]))
        if normalize:
            images = (images - self.mean) / self.std

        self.encoder(images * masks)
        features = self.encoder.retained_layer(self.layer)
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(len(images), *self.feature_shape)
        return features

    def map(  # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any,
    ) -> data.TensorDataset:
        """Override `Encoder.map`, but change defaults for single image."""
        kwargs.setdefault('mask', False)
        kwargs.setdefault('image_index', 0)
        return super().map(*args, **kwargs)

    def properties(self) -> serialize.Properties:
        """Override `Serializable.properties`."""
        return {'config': self.config, **self.kwargs}

    @staticmethod
    def configs() -> Mapping[str, SpatialConvEncoderConfig]:
        """Return the support configs mapped to their names."""
        return {
            'resnet18': (models.resnet18, ('layer4',), 49, 512),
        }


PyramidConvEncoderConfig = Tuple[ClassifierFactory, ClassifierLayers,
                                 ClassifierFeatureSize]


class PyramidConvEncoder(Encoder, serialize.SerializableModule):
    """Encode images at multiple resolutions into a single vector.

    Images are fed to a pretrained image classifier trained on ImageNet.
    The convolutional features from each layer are then masked using the
    downsampled mask, pooled, and stacked to create a feature vector.
    """

    def __init__(self, config: str = 'resnet50', **kwargs: Any):
        """Initialize the encoder.

        Keyword arguments are forwarded to the constructor of the underlying
        classifier model.

        Args:
            config (str, optional): The encoder config to use.
                See `PyramidConvEncoder.configs` for options.
                Defaults to 'resnet50'.

        """
        super().__init__()

        configs = PyramidConvEncoder.configs()
        if config not in configs:
            raise ValueError(f'encoder not supported: {config}')

        self.config = config
        self.kwargs = kwargs
        self.kwargs.setdefault('pretrained', True)

        factory, layers, feature_size = configs[config]
        self.encoder = nethook.InstrumentedModel(factory(**self.kwargs))
        self.encoder.retain_layers(layers)
        self.encoder.eval()

        self.layers = layers
        self.feature_shape = (feature_size,)

        # We will manually normalize input images. This just makes life easier.
        mean, std = renormalize.OFFSET_SCALE['imagenet']
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self,
                images: torch.Tensor,
                masks: Optional[torch.Tensor] = None,
                normalize: bool = True,
                **_: Any) -> torch.Tensor:
        """Construct pyramid features."""
        if masks is None:
            masks = images.new_ones((len(images), 1, *images.shape[2:]))
        if normalize:
            images = (images - self.mean) / self.std

        # Feed images to encoder, letting nethook record layer activations.
        self.encoder(images)
        features = self.encoder.retained_features(clear=True).values()

        # Mask the features at each level of the pyramid.
        masked = []
        for fs in features:
            ms = functional.interpolate(masks,
                                        size=fs.shape[-2:],
                                        mode='bilinear',
                                        align_corners=False)

            # Normalize the masks so they look more like attention. If any
            # of them are all zeros, we'll end up with divide-by-zero errors.
            zeros = torch.zeros_like(ms)
            valid = ~ms.isclose(zeros).all(dim=-1).all(dim=-1).view(-1)
            indices = valid.nonzero().squeeze()
            ms[indices] /= ms[indices].sum(dim=(-1, -2), keepdim=True)

            # Pool features and move on.
            mfs = fs.mul(ms).sum(dim=(-1, -2))
            masked.append(mfs)

        return torch.cat(masked, dim=-1)

    def properties(self) -> serialize.Properties:
        """Override `Serializable.properties`."""
        return {'config': self.config, **self.kwargs}

    @staticmethod
    def configs() -> Mapping[str, PyramidConvEncoderConfig]:
        """Return the support configs mapped to their names."""
        return {
            'alexnet': (
                models.alexnet,
                ('features.0', 'features.3', 'features.6', 'features.8',
                 'features.10'),
                1152,
            ),
            'resnet18': (
                models.resnet18,
                ('conv1', 'layer1', 'layer2', 'layer3', 'layer4'),
                1024,
            ),
            'resnet50': (
                models.resnet50,
                ('conv1', 'layer1', 'layer2', 'layer3', 'layer4'),
                3904,
            ),
            'resnet101': (
                models.resnet101,
                ('conv1', 'layer1', 'layer2', 'layer3', 'layer4'),
                3904,
            ),
        }


def parse(key: str) -> Type[Encoder]:
    """Parse the string key into an encoder type."""
    return {
        Type.__name__: Type
        for Type in (SpatialConvEncoder, PyramidConvEncoder)
    }[key]


def key(encoder: Encoder) -> str:
    """Return the key for the given encoder."""
    return type(encoder).__name__


KIND_SPATIAL = 'spatial'
KIND_PYRAMID = 'pyramid'


def encoder(kind: str = KIND_PYRAMID, **kwargs: Any) -> Encoder:
    """Create an encoder.

    Keyword arguments passed to constructor.

    Args:
        kind (str, optional): The kind of encoder to load. Can be
            'pyramid', 'spatial', or exact type name. Defaults to 'pyramid'.

    Returns:
        Encoder: The instantiated encoder.

    """
    encoder_t: Type[Encoder]
    if kind == KIND_SPATIAL:
        encoder_t = SpatialConvEncoder
    elif kind == KIND_PYRAMID:
        encoder_t = PyramidConvEncoder
    else:
        encoder_t = parse(kind)
    return encoder_t(**kwargs)
