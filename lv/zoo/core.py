"""Core tools for interacting with the zoo."""
import dataclasses
import pathlib
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

from lv.utils.typing import Device, Layer, PathLike

import torch
from torch import hub, nn
from torch.utils import data


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
             map_location: Optional[Device] = None,
             **kwargs: Any) -> Tuple[nn.Sequential, Iterable[Layer]]:
        """Load the model from the given path.

        Args:
            path (Optional[PathLike], optional): Path to the pretrained model
                weights. If not set, model will be initialized to whatever the
                factory function returns. If set and path does not exist but
                URL field is set, weights will be downloaded to this path.
            map_location (Optional[Device], optional): Passed to `torch.load`,
                effectively sending all model weights to this device at load
                time. Defaults to None.

        Returns:
            Tuple[nn.Sequential, Iterable[Layer]]: The loaded model.

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

        return model, layers


ModelConfigs = Mapping[str, Mapping[str, ModelConfig]]


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


DatasetConfigs = Mapping[str, DatasetConfig]
