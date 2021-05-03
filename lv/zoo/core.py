"""Core tools for interacting with the zoo."""
import dataclasses
import pathlib
import zipfile
from typing import (Any, Callable, Iterable, Mapping, Optional, OrderedDict,
                    Tuple)

from lv.utils.typing import Device, Layer, PathLike

import torch
from torch import hub, nn
from torch.utils import data

TransformWeights = Callable[[Any], OrderedDict[str, torch.Tensor]]


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
                 load_weights: bool = True,
                 transform_weights: Optional[TransformWeights] = None,
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
            load_weights (bool, optional): If True, attempt to load
                pretrained weights. Otherwise, model will be immediately
                returned after instantiation from the factory. Set this to
                False if the model returned by the factory has already loaded
                pretrained weights.
            transform_weights (Optional[TransformWeights], optional): Call
                this function on weights loaded from disk before passing them
                to the model.

        """
        self.factory = factory
        self.defaults = defaults

        self.url = url
        self.layers = layers
        self.load_weights = load_weights
        self.transform_weights = transform_weights

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
                path.parent.mkdir(exist_ok=True, parents=True)
                hub.download_url_to_file(self.url, path)
            if not path.exists():
                raise FileNotFoundError(f'model path not found: {path}')

            state_dict = torch.load(path, map_location=map_location)
            if self.transform_weights is not None:
                state_dict = self.transform_weights(state_dict)

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

    def __init__(self,
                 factory: Callable[..., data.Dataset],
                 url: Optional[str] = None,
                 requires_path: bool = True,
                 **defaults: Any):
        """Initialize the configuration.

        The keyword arguments are treated as defaults for the dataset.

        Args:
            factory (Callable[..., data.Dataset]): Factory function that
                creates the dataset from a path and arbitrary kwargs.
            url (Optional[str], optional): URL to a zip file containing the
                dataset. Will attempt to download and unzip files from this
                URL at load time. Defaults to None.
            requires_path (bool, optional): If set, dataset factory requires
                path as input and it will be passed as the first argument.
                Otherwise, path will not be passed. Defaults to True.

        """
        self.factory = factory
        self.url = url
        self.requires_path = requires_path
        self.defaults = defaults

    def load(self, path: Optional[PathLike] = None, **kwargs) -> data.Dataset:
        """Load the dataset from the given path.

        Args:
            path (Optional[PathLike], optional): Dataset path. If it does
                not exist but URL is set on the config, dataset will be
                downloaded.

        Returns:
            data.Dataset: The loaded dataset.

        """
        for key, default in self.defaults.items():
            kwargs.setdefault(key, default)

        # If no path is set, no point in doing anything else!
        if path is None:
            if self.requires_path:
                raise ValueError('dataset requires path, but none given')
            return self.factory(**kwargs)

        # Otherwise, handle URL if it is set and pass path as an arg.
        path = pathlib.Path(path)
        if not path.exists() and self.url is not None:
            file = path.parent / self.url.split('/')[-1]
            hub.download_url_to_file(self.url, file)
            with zipfile.ZipFile(file, 'r') as handle:
                handle.extractall(path)

        if not path.exists():
            raise FileNotFoundError(f'dataset path does not exist: {path}')

        return self.factory(path, **kwargs)


DatasetConfigs = Mapping[str, DatasetConfig]
