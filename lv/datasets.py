"""Datasets for dissection results and annotations."""
import collections
import csv
import pathlib
from typing import (Any, Callable, Iterable, NamedTuple, Optional, Sequence,
                    Union)

from lv.typing import PathLike

import torch
import tqdm
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms

Transform = Callable[[Image.Image], torch.Tensor]

DEFAULT_TRANSFORM = transforms.ToTensor()


class TopImages(NamedTuple):
    """Top images for a unit."""

    layer: str
    unit: str
    images: torch.Tensor


class TopImagesDataset(data.Dataset[TopImages]):
    """Top-activating images for invidual units."""

    def __init__(self,
                 root: PathLike,
                 layers: Optional[Iterable[str]] = None,
                 transform: Transform = DEFAULT_TRANSFORM,
                 cache: Union[bool, str, torch.device] = False,
                 display_progress: bool = True,
                 validate_top_image_counts: bool = True):
        """Initialize the dataset.

        Args:
            root (PathLike): Root directory for the dataset. See
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
        root = pathlib.Path(root)
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
                    raise ValueError(f'differing top image counts: {counts}; '
                                     'set validate_top_image_counts=False '
                                     'to ignore')

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


DEFAULT_LAYER_COLUMN = 'layer'
DEFAULT_UNIT_COLUMN = 'unit'
DEFAULT_ANNOTATION_COLUMN = 'summary'
DEFAULT_ANNOTATIONS_FILE_NAME = 'annotations.csv'


class AnnotatedTopImages(NamedTuple):
    """Top images and annotation for a unit."""

    layer: str
    unit: str
    images: torch.Tensor
    annotations: Sequence[str]


class AnnotatedTopImagesDataset(data.Dataset[AnnotatedTopImages]):
    """Same as TopImagesDataset, but each unit also has annotations."""

    def __init__(self,
                 root: PathLike,
                 *args: Any,
                 annotations_csv_file: Optional[PathLike] = None,
                 layer_column: str = DEFAULT_LAYER_COLUMN,
                 unit_column: str = DEFAULT_UNIT_COLUMN,
                 annotation_column: str = DEFAULT_ANNOTATION_COLUMN,
                 validate_top_image_annotated: bool = True,
                 validate_top_image_annotation_counts: bool = True,
                 **kwargs: Any):
        """Initialize the dataset.

        All *args and **kwargs are forwarded to TopImagesDataset.

        Args:
            annotations_csv_file (Optional[PathLike], optional): Path to
                annotations CSV file.
                Defaults to `root / DEFAULT_ANNOTATIONS_FILE_NAME`.
            layer_column (str, optional): CSV column containing layer name.
                Defaults to `DEFAULT_LAYER_COLUMN`.
            unit_column (str, optional): CSV column containing unit name.
                Defaults to `DEFAULT_UNIT_COLUMN`.
            annotation_column (str, optional): CSV column containing
                annotation. Defaults to `DEFAULT_ANNOTATION_COLUMN`.
            validate_top_image_annotated (bool, optional): If set, validate all
                units have at least one annotation. Defaults to True.
            validate_top_image_annotation_counts (bool, optional): If set,
                validate all annotated units have the same number of
                annotations. Defaults to True.

        Raises:
            FileNotFoundError: If annotations CSV file is not found.
            KeyError: If CSV is missing layer, unit, or annotation column.
            ValueError: If either validate flag is set and validation fails.

        """
        root = pathlib.Path(root)
        if annotations_csv_file is None:
            annotations_csv_file = root / DEFAULT_ANNOTATIONS_FILE_NAME

        annotations_csv_file = pathlib.Path(annotations_csv_file)
        if not annotations_csv_file.is_file():
            raise FileNotFoundError(
                f'annotations_csv_file not found: {annotations_csv_file}')

        self.top_images_dataset = TopImagesDataset(root, *args, **kwargs)

        with annotations_csv_file.open('r') as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames is not None, 'null columns?'
            fields = set(reader.fieldnames)
            rows = tuple(reader)

        for column in (layer_column, unit_column, annotation_column):
            if column not in fields:
                raise KeyError(f'annotations csv missing column: {column}')

        annotations = collections.defaultdict(list)
        for row in rows:
            layer = row[layer_column]
            unit = row[unit_column]
            annotation = row[annotation_column]
            annotations[layer, unit].append(annotation)
        self.annotations = {k: tuple(vs) for k, vs in annotations.items()}

        if validate_top_image_annotated:
            n_annotated_units = len(self.annotations)
            n_units = len(self.top_images_dataset)
            if n_annotated_units != n_units:
                raise ValueError(f'only {n_annotated_units} of {n_units} '
                                 'have annotations; set '
                                 'validate_top_image_annotated=False '
                                 'to ignore')

        if validate_top_image_annotation_counts:
            counts = {len(vs) for vs in self.annotations.values()}
            if len(counts) != 1:
                raise ValueError(
                    f'differing annotation counts: {counts}; '
                    'set validate_top_image_annotation_counts=False '
                    'to ignore')

    def __getitem__(self, index: int) -> AnnotatedTopImages:
        """Return the annotated top images.

        Args:
            index (int): Sample index.

        Returns:
            AnnotatedTopImages: The sample.

        """
        top_images = self.top_images_dataset[index]
        layer = top_images.layer
        unit = top_images.unit
        annotations = self.annotations.get((layer, unit), ())
        return AnnotatedTopImages(layer=top_images.layer,
                                  unit=top_images.unit,
                                  images=top_images.images,
                                  annotations=annotations)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.top_images_dataset)
