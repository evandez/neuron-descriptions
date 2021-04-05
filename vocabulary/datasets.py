"""Datasets for dissection results and annotations."""
import collections
import pathlib
from typing import Callable, Iterable, NamedTuple, Optional, Union

import torch
import tqdm
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms

Transform = Callable[[Image.Image], torch.Tensor]

DEFAULT_TRANSFORM = transforms.ToTensor()


class TopImages(NamedTuple):
    """Single neuron activation sample."""

    layer: str
    unit: str
    images: torch.Tensor


class TopImagesDataset(data.Dataset[TopImages]):
    """Top-activating images for invidual units."""

    def __init__(self,
                 root: pathlib.Path,
                 layers: Optional[Iterable[str]] = None,
                 transform: Transform = DEFAULT_TRANSFORM,
                 cache: Union[bool, str, torch.device] = False,
                 display_progress: bool = True,
                 validate_top_image_counts: bool = True):
        """Initialize the dataset.

        Args:
            root (pathlib.Path): Root directory for the dataset. See
                `dissection.dissect` function for expected format.
            layers (Optional[Iterable[str]], optional): The layers to load.
                Layer data is assumed to be a subdirectory of the root.
                By default, all subdirectories of root are treated as layers.
            transform (Transform, optional): Call this function on every image
                when it is read and use the returned result. Note that this
                function MUST return a tensor. Defaults to DEFAULT_TRANSFORM.
            cache (Union[bool, str, torch.device], optional): If set, read all
                images from disk into memory. If a device or string is
                specified, images are sent to this device. Defaults to False.
            display_progress (bool, optional): Show the progress bar when
                reading images into menu. Has no effect if `cache` is not set.
                Defaults to True.
            validate_top_image_counts (bool, optional): Check that all units
                have the same number of top images. The methods on this
                class will still work even if they don't, but some torch tools
                such as `torch.utils.DataLoader` will not out of the box.
                Defaults to True.

        Raises:
            FileNotFoundError: If root directory does not exist.
            ValueError: If no layers found or provided, or if units have
                different number of top images.

        """
        if not root.is_dir():
            raise FileNotFoundError(f'root directory not found: {root}')
        if layers is None:
            layers = [str(f.name) for f in root.iterdir() if f.is_dir()]
        if not layers:
            raise ValueError('no layers given and root has no subdirectories')

        self.root = root
        self.layers = tuple(sorted(layers))
        self.transform = transform
        self.cache = cache

        self.datasets_by_layer = {}
        for layer in self.layers:
            self.datasets_by_layer[layer] = datasets.ImageFolder(
                root / layer, transform=transform)

        self.samples = []
        for layer in self.layers:
            dataset = self.datasets_by_layer[layer]

            samples_by_unit = collections.defaultdict(list)
            for index, (_, unit_index) in enumerate(dataset.samples):
                unit = dataset.classes[unit_index]
                samples_by_unit[unit].append(index)

            for unit, indices in sorted(samples_by_unit.items(),
                                        key=lambda kv: kv[0]):
                sample = (layer, unit, indices)
                self.samples.append(sample)

            if validate_top_image_counts:
                counts = {len(indices) for _, _, indices in self.samples}
                if len(counts) != 1:
                    raise ValueError(f'differing top image counts: {counts}')

        self.images = None
        if cache:
            device = cache if not isinstance(cache, bool) else None

            samples = self.samples
            if display_progress:
                samples = tqdm.tqdm(samples,
                                    desc=f'caching images from {root.name}')

            self.images = []
            for layer, _, indices in samples:
                dataset = self.datasets_by_layer[layer]
                images = torch.stack([dataset[i][0] for i in indices])
                images = images.to(device)
                self.images.append(images)

    def __getitem__(self, index: int) -> TopImages:
        """Return the top images.

        Args:
            index (int): Sample index.

        Returns:
            TopImages: The sample.

        """
        layer, unit, indices = self.samples[index]
        if self.images is not None:
            images = self.images[index]
        else:
            dataset = self.datasets_by_layer[layer]
            images = torch.stack([dataset[i][0] for i in indices])
        return TopImages(layer=layer, unit=unit, images=images)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
