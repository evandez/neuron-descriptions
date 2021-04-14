"""Defines dissection configurations."""
import dataclasses
import pathlib
from typing import Any, Callable, Mapping, Optional, Type, TypeVar

from lv.typing import PathLike
from third_party.netdissect import renormalize

from torch.utils import data
from torchvision import datasets, transforms

DatasetConfigT = TypeVar('DatasetConfigT', bound='DatasetConfig')


@dataclasses.dataclass(frozen=True)
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
