"""Utilities for training models."""
import pathlib
from typing import Any, Sequence, Sized, Tuple, cast

from src.utils.typing import PathLike

from torch.utils import data
from torchvision import datasets
from tqdm import tqdm


class EarlyStopping:
    """Observes a numerical value and determines when it has not improved."""

    def __init__(self, patience: int = 4, decreasing: bool = True):
        """Initialize the early stopping tracker.

        Args:
            patience (int, optional): Allow tracked value to not improve over
                its best value this many times. Defaults to 4.
            decreasing (bool, optional): If True, the tracked value "improves"
                if it decreases. If False, it "improves" if it increases.
                Defaults to True.

        """
        self.patience = patience
        self.decreasing = decreasing
        self.best = float('inf') if decreasing else float('-inf')
        self.num_bad = 0

    def __call__(self, value: float) -> bool:
        """Considers the new tracked value and decides whether to stop.

        Args:
            value (float): The new tracked value.

        Returns:
            bool: True if patience has been exceeded.

        """
        improved = self.decreasing and value < self.best
        improved |= not self.decreasing and value > self.best
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1

        return self.num_bad > self.patience

    @property
    def improved(self) -> bool:
        """Check if the running value just improved."""
        return self.num_bad == 0


def random_split(dataset: data.Dataset,
                 hold_out: float = .1) -> Tuple[data.Subset, data.Subset]:
    """Randomly split the dataset into a train and val set.

    Args:
        dataset (data.Dataset): The full dataset.
        hold_out (float, optional): Fraction of data to hold out for the
            val set. Defaults to .1.

    Returns:
        Tuple[data.Subset, data.Subset]: The train and val sets.

    """
    if hold_out <= 0 or hold_out >= 1:
        raise ValueError(f'hold_out must be in (0, 1), got {hold_out}')

    size = len(cast(Sized, dataset))
    val_size = int(hold_out * size)
    train_size = size - val_size

    for name, size in (('train', train_size), ('val', val_size)):
        if size == 0:
            raise ValueError(f'hold_out={hold_out} causes {name} set size '
                             'to be zero')

    splits = data.random_split(dataset, (train_size, val_size))
    assert len(splits) == 2
    train, val = splits
    return train, val


def fixed_split(dataset: data.Dataset,
                indices: Sequence[int]) -> Tuple[data.Subset, data.Subset]:
    """Split dataset on the given indices.

    Args:
        dataset (data.Dataset): The dataset to split.
        indices (Sequence[int]): Indices comprising the right split.

    Returns:
        Tuple[data.Subset, data.Subseet]: The subset *not* for the indices,
            followed by the subset *for* the indices.

    """
    size = len(cast(Sized, dataset))
    for index in indices:
        if index < 0 or index >= size:
            raise IndexError(f'dataset index out of bounds: {index}')

    others = sorted(set(range(size)) - set(indices))
    if not others:
        raise ValueError('indices cover entire dataset; nothing to split!')

    return data.Subset(dataset, others), data.Subset(dataset, indices)


# TODO(evandez): This really isn't a very elegant solution to the threading
# problem of ImageFolder, as it loads the images in serial and this is very
# slow for any dataset worth its muster. Better to figure out why threading
# causes so many problems and fix that.
class PreloadedImageFolder(data.Dataset):
    """An ImageFolder that preloads all the images."""

    def __init__(self,
                 root: PathLike,
                 *args: Any,
                 display_progress: bool = True,
                 **kwargs: Any):
        """Preload the dataset.

        The *args and **kwargs are forwarded to the superclass constructor.

        Args:
            root (PathLike): Dataset root.
            display_progress (bool, optional): Display progress as dataset
                is being loaded.

        """
        self.dataset = datasets.ImageFolder(str(root), *args, **kwargs)

        self.cached_images = []
        self.cached_labels = []

        indices = range(len(self.dataset))
        if display_progress:
            root = pathlib.Path(root)
            indices = tqdm(indices,
                           desc=f'preload {root.parent.name}/{root.name}')

        for index in indices:
            image, label = self.dataset[index]
            self.cached_images.append(image)
            self.cached_labels.append(label)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Access image and label from cache.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[Any, Any]: The image and label.

        """
        return self.cached_images[index], self.cached_labels[index]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)
