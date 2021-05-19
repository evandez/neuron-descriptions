"""Utilities for training models."""
import pathlib
from typing import Any, cast

from lv.utils.typing import PathLike

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


class PreloadedImageFolder(datasets.ImageFolder):
    """An ImageFolder that preloads all the images."""

    def __init__(self,
                 root: PathLike,
                 *args: Any,
                 num_workers: int = 0,
                 display_progress: bool = True,
                 **kwargs: Any):
        """Preload the dataset.

        The *args and **kwargs are forwarded to the superclass constructor.

        Args:
            root (PathLike): Dataset root.
            num_workers (data.Dataset): Passed to `data.DataLoader` while
                images are being cahced.
            display_progress (bool, optional): Display progress as dataset
                is being loaded.

        """
        super().__init__(str(root), *args, **kwargs)

        self.cached_images = []
        self.cached_labels = []

        loader = data.DataLoader(cast(data.Dataset, self),
                                 num_workers=num_workers)
        if display_progress:
            root = pathlib.Path(root)
            loader = tqdm(loader,
                          desc=f'preload {root.parent.name}/{root.name}')

        for images, labels in loader:
            self.cached_images.extend(images)
            self.cached_labels.extend(labels)

    def __getitem__(self, index):
        """Access image and label from cache."""
        return self.cached_images[index], self.cached_labels[index]
