"""Tools for accessing models and datasets stored remotely."""
import dataclasses
import pathlib
import tempfile
import zipfile
from typing import Any, Callable, Mapping, Optional, OrderedDict

from src.utils import env
from src.utils.typing import Device, PathLike

import torch
from torch import hub, nn
from torch.utils import data

HOST = 'http://milan.csail.mit.edu'

TransformWeights = Callable[[Any], OrderedDict[str, torch.Tensor]]
ModelFactory = Callable[..., nn.Module]


@dataclasses.dataclass
class ModelConfig:
    """Model configuration.

    This dataclass wraps a factory for the model and, optionally, the names
    of its layers, the url containing its pretrained weights, and default
    keyword arguments to be passed to the factory, all of which can be
    overridden at construction.
    """

    def __init__(self,
                 factory: ModelFactory,
                 url: Optional[str] = None,
                 requires_path: bool = False,
                 load_weights: bool = True,
                 transform_weights: Optional[TransformWeights] = None,
                 **defaults: Any):
        """Initialize  the configuration.

        The keyword arguments are treated as defaults for the model.

        Args:
            factory (ModelFactory): Factory function that
                creates a model from arbitrary keyword arguments.
            url (Optional[str], optional): URL hosting pretrained weights.
                If set and path provided to `load` does not exist, weights
                will be downloaded. Defaults to None.
            requires_path (bool, optional): If True, path argument is required
                and will be forwarded to the factory as the first argument.
                Defaults to False.
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
        self.requires_path = requires_path
        self.load_weights = load_weights
        self.transform_weights = transform_weights

    def load(self,
             path: Optional[PathLike] = None,
             factory: Optional[ModelFactory] = None,
             load_weights: Optional[bool] = None,
             map_location: Optional[Device] = None,
             **kwargs: Any) -> nn.Module:
        """Load the model from the given path.

        Args:
            path (Optional[PathLike], optional): Path to the pretrained model
                weights. If not set, model will be initialized to whatever the
                factory function returns. If set and path does not exist but
                URL field is set, weights will be downloaded to this path.
            factory (Optional[ModelFactory], optional): Override for config
                default factory. Defaults to None.
            load_weights (Optional[bool], optional): Override for config
                on whether weights should be loaded. Defaults to None.
            map_location (Optional[Device], optional): Passed to `torch.load`,
                effectively sending all model weights to this device at load
                time. Defaults to None.

        Returns:
            nn.Module: The loaded model.

        """
        if path is None and self.requires_path:
            raise ValueError('model requires path, but none given')

        # Set defaults.
        if factory is None:
            factory = self.factory
        if load_weights is None:
            load_weights = self.load_weights

        # Set factory fn defaults.
        for key, default in self.defaults.items():
            kwargs.setdefault(key, default)

        # If necessary, try to download model weights from the URL.
        if path is not None and (load_weights or self.requires_path):
            path = pathlib.Path(path)
            if not path.exists() and self.url is not None:
                path.parent.mkdir(exist_ok=True, parents=True)
                hub.download_url_to_file(self.url, path)
            if not path.exists():
                raise FileNotFoundError(f'model path not found: {path}')

        # Create the model, forwarding path if needed.
        if self.requires_path:
            model = factory(path, **kwargs)
        else:
            model = factory(**kwargs)

        # Explicitly load the weights if needed.
        if path is not None and load_weights:
            state_dict = torch.load(path, map_location=map_location)
            if self.transform_weights is not None:
                state_dict = self.transform_weights(state_dict)

            model.load_state_dict(state_dict)

        return model.eval()


class ModelHub:
    """A model hub."""

    configs: Mapping[str, ModelConfig]

    def __init__(self, **configs: ModelConfig):
        """Initialize the hub with the given configs."""
        self.configs = configs

    def load(self,
             name: str,
             path: Optional[PathLike] = None,
             **kwargs: Any) -> nn.Module:
        """Load the model trained on the given dataset.

        Args:
            name (str): Name of the model.
            path (Optional[PathLike], optional): Path to the model weights.
                If not set, defaults to `<project model dir>/{name}.pth`.
                If path does not exist but `url` is set on the model config,
                weights will be downloaded from URL to the path.

        Raises:
            KeyError: If no model with given name exists.

        Returns:
            Model: The loaded model and its config.

        """
        if name not in self.configs:
            raise KeyError(f'no such model in hub: {name}')
        config = self.configs[name]

        if path is None:
            path = env.models_dir() / f'{name}.pth'

        model = config.load(path, **kwargs)
        return model


DatasetFactory = Callable[..., data.Dataset]


class DatasetConfig:
    """Dataset configuration.

    For the most part, this class just wraps a factory for the dataset. It also
    supports setting defaults arguments. These defaults can be overwritten by
    the keyword arguments provided to `zoo.dataset`.
    """

    def __init__(self,
                 factory: DatasetFactory,
                 url: Optional[str] = None,
                 requires_path: bool = True,
                 **defaults: Any):
        """Initialize the configuration.

        The keyword arguments are treated as defaults for the dataset.

        Args:
            factory (DatasetFactory): Factory function that creates the dataset
                from a path and arbitrary kwargs.
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

    def load(self,
             path: Optional[PathLike] = None,
             factory: Optional[DatasetFactory] = None,
             **kwargs: Any) -> data.Dataset:
        """Load the dataset from the given path.

        Args:
            path (Optional[PathLike], optional): Dataset path. If it does
                not exist but URL is set on the config, dataset will be
                downloaded.
            factory (Optional[DatasetFactory]): Override for config default
                factory.

        Returns:
            data.Dataset: The loaded dataset.

        """
        if factory is None:
            factory = self.factory

        for key, default in self.defaults.items():
            kwargs.setdefault(key, default)

        # If no path is set, no point in doing anything else!
        if path is None:
            if self.requires_path:
                raise ValueError('dataset requires path, but none given')
            return factory(**kwargs)

        # Otherwise, handle URL if it is set and pass path as an arg.
        path = pathlib.Path(path)
        if not path.exists() and self.url is not None:
            path.mkdir(parents=True)
            with tempfile.TemporaryDirectory() as tempdir:
                file = pathlib.Path(tempdir) / self.url.split('/')[-1]
                hub.download_url_to_file(self.url, file)
                with zipfile.ZipFile(file, 'r') as handle:
                    handle.extractall(path)

        if not path.exists():
            raise FileNotFoundError(f'dataset path does not exist: {path}')

        return factory(path, **kwargs)


class DatasetHub:
    """A dataset hub."""

    def __init__(self, **configs: DatasetConfig):
        """Initialize the hub with the given configs."""
        self.configs = configs

    def load(self,
             name: str,
             path: Optional[PathLike] = None,
             **kwargs: Any) -> data.Dataset:
        """Load the dataset with the given name.

        Args:
            name (str): Dataset configuration name.
            path (Optional[PathLike], optional): Path to dataset. Defaults to
                project default (see `src.utils.env`).

        Returns:
            data.Dataset: The loaded dataset.

        Raises:
            KeyError: If no dataset with the given name exists.

        """
        if name not in self.configs:
            raise KeyError(f'no such dataset in hub: {name}')
        config = self.configs[name]

        if path is None and config.requires_path:
            path = env.data_dir() / name

        dataset = config.load(path=path, **kwargs)

        return dataset

    def load_all(self,
                 name: str,
                 *others: str,
                 path: Optional[PathLike] = None,
                 **kwargs: Any) -> data.Dataset:
        """Load each dataset and concatenate them.

        Args:
            name (str): First dataset configuration name. Must specify at least
                this argument.
            path (Optional[PathLike], optional): Root path for all datasets.
                Individual paths will be computed by appending dataset name to
                this path. Defaults to project default.

        Returns:
            data.Dataset: All datasets concatenated into one.

        """
        if path is None:
            path = env.data_dir()
        concated = self.load(name, path=pathlib.Path(path) / name, **kwargs)
        for other in others:
            concated += self.load(other,
                                  path=pathlib.Path(path) / other,
                                  **kwargs)
        return concated
