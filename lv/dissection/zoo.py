"""Defines dissection configurations."""
import dataclasses
import pathlib
from typing import (Any, Callable, Iterable, Mapping, Optional, Tuple, Type,
                    TypeVar, Union)

from lv.ext.torchvision import models
from lv.typing import Layer, PathLike
from third_party.netdissect import renormalize

import torch
from torch import hub, nn
from torch.utils import data
from torchvision import datasets, transforms

ModelConfigT = TypeVar('ModelConfigT', bound='ModelConfig')
ModelConfigsT = Mapping[str, Mapping[str, ModelConfigT]]


@dataclasses.dataclass
class ModelConfig:
    """Model configuration.

    This dataclass wraps a factory for the model and, optionally, the names
    of its layers, the url containing its pretrained weights, and default
    keyword arguments to be passed to the factory, all of which can be
    overridden at construction.
    """

    def __init__(self,
                 factory: Callable[..., nn.Sequential],
                 layers: Optional[Iterable[Layer]] = None,
                 url: Optional[str] = None,
                 generative: bool = False,
                 load_weights: bool = True,
                 **defaults: Any):
        """Initialize  the configuration.

        The keyword arguments are treated as defaults for the model.

        Args:
            factory (Callable[..., nn.Sequential]): Factory function that
                creates a model from arbitrary keyword arguments.
            layers (Optional[Iterable[Layer]], optional): Layers to return
                when model is instantiated. By default, set to the keys
                returned by `model.named_children()`.
            url (Optional[str], optional): URL hosting pretrained weights.
                If set and path provided to `load` does not exist, weights
                will be downloaded. Defaults to None.
            generative (bool, optional): Set to True if this is a generative
                model of images. Defaults to False.
            load_weights (bool, optional): If True, attempt to load
                pretrained weights. Otherwise, model will be immediately
                returned after instantiation from the factory. Set this to
                False if the model returned by the factory has already loaded
                pretrained weights.

        """
        self.factory = factory
        self.defaults = defaults

        self.url = url
        self.generative = generative
        self.layers = layers
        self.load_weights = load_weights

    def load(self,
             path: Optional[PathLike] = None,
             map_location: Optional[Union[str, torch.device]] = None,
             **kwargs: Any) -> nn.Sequential:
        """Load the model from the given path.

        Args:
            path (Optional[PathLike], optional): Path to the pretrained model
                weights. If not set, model will be initialized to whatever the
                factory function returns. If set and path does not exist but
                URL field is set, weights will be downloaded to this path.
            map_location (Optional[Union[str, torch.device]], optional): Passed
                to `torch.load`, effectively sending all model weights to this
                device at load time. Defaults to None.

        Returns:
            nn.Sequential: The loaded model.

        """
        for key, default in self.defaults.items():
            kwargs.setdefault(key, default)

        model = self.factory(**kwargs)

        if path is not None and self.load_weights:
            path = pathlib.Path(path)
            if not path.exists() and self.url is not None:
                hub.download_url_to_file(self.url, path)
            if not path.exists():
                raise FileNotFoundError(f'model path not found: {path}')
            state_dict = torch.load(path, map_location=map_location)
            model.load_state_dict(state_dict)

        layers = self.layers
        if layers is None:
            layers = [key for key, _ in model.named_children()]

        return model

    @classmethod
    def configs(cls: Type[ModelConfigT]) -> ModelConfigsT:
        """Return default configs."""
        return {
            'alexnet': {
                'imagenet':
                    cls(models.alexnet_seq,
                        pretrained=True,
                        load_weights=False),
            },
            'resnet18': {
                'imagenet':
                    cls(models.resnet18_seq,
                        pretrained=True,
                        load_weights=False)
            },
            'vgg16': {
                'imagenet':
                    cls(models.vgg16_seq, pretrained=True, load_weights=False)
            }
        }


ModelConfigs = Mapping[str, Mapping[str, ModelConfig]]


def model(name: str,
          dataset: str,
          path: Optional[PathLike] = None,
          configs: Optional[ModelConfigs] = None,
          **kwargs: Any) -> Tuple[nn.Sequential, ModelConfig]:
    """Load the model trained on the given dataset.

    Args:
        name (str): Name of the model.
        dataset (str): Name of the dataset.
        path (Optional[PathLike], optional): Path to the model weights.
            If not set, defaults to `.zoo/models/{name}-{dataset}.pth`.
            If path does not exist but `url` is set on the model config,
            weights will be downloaded from URL to the path.
        configs (Optional[ModelConfigs], optional): Mapping from model names
            to a mapping from dataset names to model configs. By default,
            calls `ModelConfigs.configs()`.

    Raises:
        KeyError: If no model with given name exists, or if model has no
            weights for the given dataset.

    Returns:
        Tuple[nn.Sequential, ModelConfig]: The loaded model as an
            `nn.Sequential` along with its config.

    """
    if configs is None:
        configs = ModelConfig.configs()
    if name not in configs:
        raise KeyError(f'no such model in zoo: {name}')
    if dataset not in configs[name]:
        raise KeyError(f'no {name} model for dataset: {dataset}')
    config = configs[name][dataset]

    if path is None:
        path = pathlib.Path(
            __file__).parents[2] / '.zoo/models' / f'{name}-{dataset}.pth'

    model = config.load(path, **kwargs)
    return model, config


DatasetConfigT = TypeVar('DatasetConfigT', bound='DatasetConfig')


class DatasetConfig:
    """Dataset configuration.

    For the most part, this class just wraps a factory for the dataset. It also
    supports setting defaults arguments. These defaults can be overwritten by
    the keyword arguments provided to `zoo.dataset`.
    """

    def __init__(self, factory: Callable[..., data.Dataset], **defaults: Any):
        """Initialize the configuration.

        The keyword arguments are treated as defaults for the dataset.

        Args:
            factory (Callable[..., data.Dataset]): Factory function that
                creates the dataset from a path and arbitrary kwargs.

        """
        self.factory = factory
        self.defaults = defaults

    def load(self, path: PathLike, **kwargs) -> data.Dataset:
        """Load the dataset from the given path.

        Args:
            path (PathLike): Dataset path.

        Returns:
            data.Dataset: The loaded dataset.

        """
        for key, default in self.defaults.items():
            kwargs.setdefault(key, default)
        return self.factory(path, **kwargs)

    @classmethod
    def configs(cls: Type[DatasetConfigT]) -> Mapping[str, DatasetConfigT]:
        """Return default configs."""
        # TODO(evandez): Are these the right transforms?
        return {
            'imagenet':
                cls(datasets.ImageNet,
                    split='val',
                    transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        renormalize.NORMALIZER['imagenet']
                    ])),
            'places365':
                cls(datasets.Places365,
                    split='val',
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor()
                    ])),
        }


def dataset(name: str,
            path: Optional[PathLike] = None,
            configs: Optional[Mapping[str, DatasetConfig]] = None,
            **kwargs: Any) -> data.Dataset:
    """Load the dataset with the given name.

    Args:
        name (str): Dataset configuration name. See `DATASET_CONFIGS` for
            all options.
        path (Optional[PathLike], optional): Path to dataset. Defaults to
            .zoo/datasets/{name} at the root of this repository.
        configs (Optional[Mapping[str, DatasetConfig]], optional): Mapping
            from config names to dataset configs. By default, calls
            `DatasetConfigs.configs()`.

    Returns:
        data.Dataset: The loaded dataset.

    Raises:
        KeyError: If no dataset with the given name exists.

    """
    if configs is None:
        configs = DatasetConfig.configs()
    if name not in configs:
        raise KeyError(f'no such dataset in zoo: {name}')
    config = configs[name]

    if path is None:
        path = pathlib.Path(__file__).parents[2] / '.zoo/datasets' / name

    dataset = config.load(path, **kwargs)

    return dataset
