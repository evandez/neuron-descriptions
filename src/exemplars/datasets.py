"""Dataset configs for computing exemplars.

The main function here is `dataset_hub`, which returns a mapping from dataset
name to a config specifying how to load it. The most important thing to know
is that the config takes a factory function for the dataset and arbitrary
kwargs to pass that factory. If a download URL is not specified, it expects
the dataset to live at $MILAN_DATA_DIR/dataset_name by default. See
`src/utils/hubs.py` for all the different options the configs support.
"""
import pathlib
from typing import Any, Mapping, Optional

from src import milannotations
from src.deps.netdissect import renormalize
from src.utils import hubs
from src.utils.typing import PathLike

import easydict
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

HOST = f'{hubs.HOST}/data'

KEYS = easydict.EasyDict(d=milannotations.KEYS)
KEYS.IMAGENET_SPURIOUS_TEXT = 'imagenet-spurious-text'
KEYS.IMAGENET_SPURIOUS_COLOR = 'imagenet-spurious-color'
KEYS.BIGGAN_ZS_IMAGENET = 'biggan-zs-imagenet'
KEYS.BIGGAN_ZS_PLACES365 = 'biggan-zs-places365'


class TensorDatasetOnDisk(torch.utils.data.TensorDataset):
    """Like `torch.utils.data.TensorDataset`, but tensors are pickled."""

    def __init__(self, root: PathLike, **kwargs: Any):
        """Load tensors from path and pass to `TensorDataset`.

        Args:
            root (PathLike): Root directory containing one or more .pth files
                of tensors.

        """
        loaded = []
        for child in pathlib.Path(root).iterdir():
            if not child.is_file() or not child.suffix == '.pth':
                continue
            tensors = torch.load(child, **kwargs)
            loaded.append(tensors)
        loaded = sorted(loaded,
                        key=lambda tensor: not tensor.dtype.is_floating_point)
        super().__init__(*loaded)


def default_dataset_configs(
        **others: hubs.DatasetConfig,  # Your overrides!
) -> Mapping[str, hubs.DatasetConfig]:
    """Return the default dataset configs."""
    configs = {
        KEYS.IMAGENET:
            hubs.DatasetConfig(torchvision.datasets.ImageFolder,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize(256),
                                   torchvision.transforms.CenterCrop(224),
                                   torchvision.transforms.ToTensor(),
                                   renormalize.NORMALIZER['imagenet']
                               ])),
        KEYS.PLACES365:
            hubs.DatasetConfig(torchvision.datasets.ImageFolder,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize(256),
                                   torchvision.transforms.CenterCrop(224),
                                   torchvision.transforms.ToTensor(),
                                   renormalize.NORMALIZER['imagenet'],
                               ])),
        KEYS.IMAGENET_SPURIOUS_TEXT:
            hubs.DatasetConfig(torchvision.datasets.ImageFolder,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((224, 224)),
                                   torchvision.transforms.ToTensor(),
                                   renormalize.NORMALIZER['imagenet']
                               ])),
        KEYS.IMAGENET_SPURIOUS_COLOR:
            hubs.DatasetConfig(torchvision.datasets.ImageFolder,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((224, 224)),
                                   torchvision.transforms.ToTensor(),
                                   renormalize.NORMALIZER['imagenet']
                               ])),
        KEYS.BIGGAN_ZS_IMAGENET:
            hubs.DatasetConfig(
                TensorDatasetOnDisk,
                url=f'{HOST}/{KEYS.BIGGAN_ZS_IMAGENET}.zip',
            ),
        KEYS.BIGGAN_ZS_PLACES365:
            hubs.DatasetConfig(
                TensorDatasetOnDisk,
                url=f'{HOST}/{KEYS.BIGGAN_ZS_PLACES365}.zip',
            ),
    }
    configs.update(others)
    return configs


def default_dataset_hub(**others: hubs.DatasetConfig) -> hubs.DatasetHub:
    """Return configs for all datasets used in dissection."""
    configs = default_dataset_configs(**others)
    return hubs.DatasetHub(**configs)


def load(name: str,
         configs: Optional[Mapping[str, hubs.DatasetConfig]] = None,
         **kwargs: Any) -> torch.utils.data.Dataset:
    """Load the dataset.

    Args:
        name (str): The name of the dataset.
        configs (Optional[Mapping[str, hubs.DatasetConfig]], optional): Configs
            to load from. Defaults to those returned by default_dataset_hub().

    Returns:
        torch.utils.data.Dataset: The loaded dataset.

    """
    configs = configs or {}
    hub = default_dataset_hub(**configs)
    return hub.load(name, **kwargs)
