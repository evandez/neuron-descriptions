"""Models that map images and masks to features."""
from typing import (Any, Callable, Mapping, Optional, Sequence, Tuple, Union,
                    overload)

from lv.ext.torchvision import models
from lv.utils import serialize
from lv.utils.typing import Device
from third_party.netdissect import nethook, renormalize

import torch
from torch import nn
from torch.nn import functional
from torch.utils import data
from tqdm.auto import tqdm


class Featurizer(nn.Module):
    """An abstract module mapping images (and optionally masks) to features."""

    feature_shape: Tuple[int, ...]

    @overload
    def forward(self, images: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Featurize the given images.

        Equivalent to featurizing the images where the masks are all 1s.

        Args:
            images (torch.Tensor): The images to featurizer. Should have
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

    def forward(self, images, masks=None, **kwargs):
        """Abstract forward function corresponding to both overloads."""
        raise NotImplementedError

    def map(self,
            dataset: data.Dataset,
            mask: bool = True,
            image_index: Union[int, str] = -3,
            mask_index: Union[int, str] = -2,
            batch_size: int = 128,
            num_workers: int = 0,
            device: Optional[Device] = None,
            display_progress_as: Optional[str] = 'featurize dataset',
            **kwargs: Any) -> data.TensorDataset:
        """Featurize an entire dataset.

        Keyword arguments are passed to `forward`.

        Args:
            dataset (data.Dataset): The dataset to featurize. Should return
                a sequence or mapping of values.
            mask (bool, optional): Try to read masks from batch and pass them
                to the featurizer. Setting this to False is equivalent to
                setting masks to be all 1s. Defaults to True.
            image_index (int, optional): Index of image in each dataset
                sample. Defaults to -3 to be compatible with
                AnnotatedTopImagesDataset.
            mask_index (int, optional): Index of mask in each dataset
                sample. Defaults to -2 to be compatible with
                AnnotatedTopImagesDataset.
            batch_size (int, optional): Featurize images in batches of this
                size. Defaults to 128.
            num_workers (int, optional): Number of workers for loading data.
                Defaults to 0.
            device (Optional[Device], optional): Run preprocessing on this
                device. Defaults to None.
            display_progress_as (Optional[str], optional): Show a progress bar
                with this key. Defaults to 'featurize dataset'.

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
        if display_progress_as is not None:
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
ImageFeaturizerConfig = Tuple[ClassifierFactory, ClassifierLayers,
                              ClassifierNumFeatures, ClassifierFeatureSize]


class ImageFeaturizer(Featurizer):
    """Featurizes images using convolutional features from a pretrained CNN."""

    def __init__(self, config: str = 'resnet18', **kwargs: Any):
        """Initialize the featurizer.

        Keyword arguments are forwarded to the constructor of the underlying
        classifier model.

        Args:
            config (str, optional): The featurizer config to use.
                See `ImageFeaturizer.configs` for options.
                Defaults to 'resnet18'.

        """
        super().__init__()

        configs = ImageFeaturizer.configs()
        if config not in configs:
            raise ValueError(f'featurizer not supported: {config}')

        self.config = config
        self.kwargs = kwargs
        self.kwargs.setdefault('pretrained', True)

        factory, layers, n_features, feature_size = configs[config]
        assert len(layers) == 1, 'multiple layers?'
        layer, = layers

        self.featurizer = nethook.InstrumentedModel(factory(**self.kwargs))
        self.featurizer.retain_layer(layer)
        self.featurizer.eval()

        self.layer = layer
        self.feature_shape = (n_features, feature_size)

        # We will manually normalize input images. This just makes life easier.
        mean, std = renormalize.OFFSET_SCALE['imagenet']
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images, masks=None, normalize=True, **_):
        """Featurize the images."""
        if masks is None:
            masks = images.new_ones(len(images), 1, *images.shape[2:])
        if normalize:
            images = (images - self.mean) / self.std

        self.featurizer(images)
        features = self.featurizer.retained_layer(self.layer)
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(len(images), *self.feature_shape)
        return features

    def map(self, *args, mask=False, image_index=0, **kwargs):
        """Override `Featurizer.map`, but change defaults for single image."""
        return super().map(*args, mask=mask, image_index=image_index, **kwargs)

    @staticmethod
    def configs() -> Mapping[str, ImageFeaturizerConfig]:
        """Return the support configs mapped to their names."""
        return {
            'resnet18': (models.resnet18_seq, ('layer4',), 49, 512),
        }


MaskedPyramidFeaturizerConfig = Tuple[ClassifierFactory, ClassifierLayers,
                                      ClassifierFeatureSize]


class MaskedPyramidFeaturizer(Featurizer, serialize.SerializableModule):
    """Map images and masks to a pyramid of masked convolutional features.

    Images are fed to a pretrained image classifier trained on ImageNet.
    The convolutional features from each layer are then masked using the
    downsampled mask, pooled, and stacked to create a feature vector.
    """

    def __init__(self, config: str = 'resnet18', **kwargs: Any):
        """Initialize the featurizer.

        Keyword arguments are forwarded to the constructor of the underlying
        classifier model.

        Args:
            config (str, optional): The featurizer config to use.
                See `MaskedPyramidFeaturizer.configs` for options.
                Defaults to 'resnet18'.

        """
        super().__init__()

        configs = MaskedPyramidFeaturizer.configs()
        if config not in configs:
            raise ValueError(f'featurizer not supported: {config}')

        self.config = config
        self.kwargs = kwargs
        self.kwargs.setdefault('pretrained', True)

        factory, layers, feature_size = configs[config]
        self.featurizer = nethook.InstrumentedModel(factory(**self.kwargs))
        self.featurizer.retain_layers(layers)
        self.featurizer.eval()

        self.layers = layers
        self.feature_shape = (feature_size,)

        # We will manually normalize input images. This just makes life easier.
        mean, std = renormalize.OFFSET_SCALE['imagenet']
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images, masks=None, normalize=True, **_: Any):
        """Construct masked features."""
        if masks is None:
            masks = images.new_ones(len(images), 1, *images.shape[2:])
        if normalize:
            images = (images - self.mean) / self.std

        # Feed images to featurizer, letting nethook record layer activations.
        self.featurizer(images)
        features = self.featurizer.retained_features(clear=True).values()

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

    def properties(self,
                   state_dict: bool = False,
                   **kwargs: Any) -> Mapping[str, Any]:
        """Return serializable values for the module.

        Usually, we do not need to save module parameters because the
        featurizer is pretrained. Hence `state_dict` defaults to False.

        Keyword arguments are forwarded to `torch.nn.Module.state_dict`.

        Args:
            state_dict (bool, optional): If set, include state_dict in
                properties. Defaults to False.

        Returns:
            Mapping[str, Any]: Model properties.

        """
        assert 'state_dict' not in self.kwargs, 'state_dict parameter?'
        properties = dict(super().properties(state_dict=state_dict, **kwargs))
        properties.update({'config': self.config, **self.kwargs})
        return properties

    @staticmethod
    def configs() -> Mapping[str, MaskedPyramidFeaturizerConfig]:
        """Return the support configs mapped to their names."""
        return {
            'alexnet': (
                models.alexnet_seq,
                ('conv1', 'conv2', 'conv3', 'conv4', 'conv5'),
                1152,
            ),
            'resnet18': (
                models.resnet18_seq,
                ('conv1', 'layer1', 'layer2', 'layer3', 'layer4'),
                1024,
            ),
        }


class MaskedImagePyramidFeaturizer(MaskedPyramidFeaturizer):
    """Same as MaskedPyramidFeaturizer, but images are masked, not features."""

    def forward(self, images, masks=None, **kwargs):
        """Mask the images and then compute pyramid features."""
        return super().forward(images * masks if masks is not None else images,
                               **kwargs)
