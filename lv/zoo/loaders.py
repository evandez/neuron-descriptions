"""Functions for loading models/datasets by name."""
import pathlib
from typing import Any, Mapping, Optional, Sequence, Tuple

from lv.utils import env
from lv.utils.typing import Layer, PathLike
from lv.zoo import configs, core

from torch import nn
from torch.utils import data

Model = Tuple[nn.Module, Sequence[Layer], core.ModelConfig]


def model(name: str,
          dataset: str,
          path: Optional[PathLike] = None,
          source: Optional[core.ModelConfigs] = None,
          **kwargs: Any) -> Model:
    """Load the model trained on the given dataset.

    Args:
        name (str): Name of the model.
        dataset (str): Name of the dataset.
        path (Optional[PathLike], optional): Path to the model weights.
            If not set, defaults to `<project model dir>/{name}-{dataset}.pth`.
            If path does not exist but `url` is set on the model config,
            weights will be downloaded from URL to the path.
        source (Optional[core.ModelConfigs], optional): Mapping from model
            names to a mapping from dataset names to model configs. By default,
            calls `lv.zoo.configs.models()`.

    Raises:
        KeyError: If no model with given name exists, or if model has no
            weights for the given dataset.

    Returns:
        Model: The loaded model along with its layers.

    """
    if source is None:
        source = configs.models()
    if name not in source:
        raise KeyError(f'no such model in zoo: {name}')
    if dataset not in source[name]:
        raise KeyError(f'no {name} model for dataset: {dataset}')
    config = source[name][dataset]

    if path is None:
        path = env.models_dir() / f'{name}-{dataset}.pth'

    model, layers = config.load(path, **kwargs)
    return model, layers, config


def dataset(name: str,
            path: Optional[PathLike] = None,
            source: Optional[Mapping[str, core.DatasetConfig]] = None,
            **kwargs: Any) -> data.Dataset:
    """Load the dataset with the given name.

    Args:
        name (str): Dataset configuration name. See `DATASET_CONFIGS` for
            all options.
        path (Optional[PathLike], optional): Path to dataset. Defaults to
            project default (see `lv.utils.env`).
        source (Optional[Mapping[str, DatasetConfig]], optional): Mapping
            from config names to dataset configs. By default, calls
            `lv.zoo.configs.datasets()`.

    Returns:
        data.Dataset: The loaded dataset.

    Raises:
        KeyError: If no dataset with the given name exists.

    """
    if source is None:
        source = configs.datasets()
    if name not in source:
        raise KeyError(f'no such dataset in zoo: {name}')
    config = source[name]

    if path is None and config.requires_path:
        path = env.data_dir() / name

    dataset = config.load(path=path, **kwargs)

    return dataset


def datasets(name: str,
             *others: str,
             path: Optional[PathLike] = None,
             **kwargs: Any) -> data.Dataset:
    """Load each dataset and concatenate them.

    Args:
        name (str): First dataset configuration name. Must specify at least
            this argument. See `DATASET_CONFIGS` for all options.
        path (Optional[PathLike], optional): Root path for all datasets.
            Individual paths will be computed by appending dataset name to
            this path. Defaults to project default.

    Returns:
        data.Dataset: All datasets concatenated into one.

    """
    if path is None:
        path = env.data_dir()
    concated = dataset(name, path=pathlib.Path(path) / name, **kwargs)
    for other in others:
        concated += dataset(other, path=pathlib.Path(path) / other, **kwargs)
    return concated
