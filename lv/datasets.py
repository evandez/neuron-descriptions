"""Datasets for dissection results and annotations."""
import collections
import csv
import pathlib
from typing import Any, Iterable, NamedTuple, Optional, Sequence, Union

from lv.third_party.netdissect import renormalize
from lv.utils.typing import Layer, PathLike, Unit

import numpy
import torch
from PIL import Image
from torch.utils import data
from torchvision import utils
from torchvision.transforms import functional
from tqdm.auto import tqdm


class TopImages(NamedTuple):
    """Top images for a unit."""

    layer: str
    unit: int
    images: torch.Tensor
    masks: torch.Tensor

    def as_pil_image_grid(self,
                          opacity: float = .75,
                          **kwargs: Any) -> Image.Image:
        """Pack all images into a grid and return as a PIL Image.

        Keyword arguments are forwarded to `torchvision.utils.make_grid`.

        Args:
            opacity (float, optional): Opacity for mask, with 1 meaning
                that the masked area is black, and 0 meaning that the masked
                area is shown as normal. Defaults to .75.

        Returns:
            Image.Image: Image grid containing all top images.

        """
        if opacity < 0 or opacity > 1:
            raise ValueError(f'opacity must be in [0, 1], got {opacity}')
        kwargs.setdefault('nrow', 5)
        masks = self.masks.clone().float()
        masks[masks == 0] = 1 - opacity
        images = self.images * masks
        grid = utils.make_grid(images, **kwargs)
        return functional.to_pil_image(grid)


class TopImagesDataset(data.Dataset):
    """Top-activating images for invidual units."""

    def __init__(self,
                 root: PathLike,
                 name: Optional[str] = None,
                 layers: Optional[Iterable[Layer]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 display_progress: bool = True):
        """Initialize the dataset.

        Args:
            root (PathLike): Root directory for the dataset. See
                `dissection.dissect` function for expected format.
            name (Optional[str], optional): Human-readable name for this
                dataset. Defaults to last two components of root directory.
            layers (Optional[Iterable[Layer]], optional): The layers to load.
                Layer data is assumed to be a subdirectory of the root.
                By default, all subdirectories of root are treated as layers.
            device (Optional[Union[str, torch.device]], optional): Send all
                tensors to this device.
            display_progress (bool, optional): Show a progress
                bar when reading images into menu. Defaults to True.

        Raises:
            FileNotFoundError: If root directory does not exist or if layer
                directory is missing images or masks.
            ValueError: If no layers found or provided, or if units have
                different number of top images.

        """
        root = pathlib.Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f'root directory not found: {root}')

        if layers is None:
            layers = [f.name for f in root.iterdir() if f.is_dir()]
        if not layers:
            raise ValueError('no layers given and root has no subdirectories')

        if name is None:
            name = f'{root.parent.name}/{root.name}'

        self.root = root
        self.name = name
        self.layers = layers = tuple(sorted(str(layer) for layer in layers))
        self.device = device

        progress = layers
        if display_progress is not None:
            progress = tqdm(progress,
                            desc=f'load {root.parent.name}/{root.name}')

        self.images_by_layer = {}
        self.masks_by_layer = {}
        renormalizer = renormalize.renormalizer(source='byte', target='pt')
        for layer in progress:
            images_file = root / str(layer) / 'images.npy'
            masks_file = root / str(layer) / 'masks.npy'
            for file in (images_file, masks_file):
                if not file.exists():
                    raise FileNotFoundError(f'{layer} is missing {file.name}')

            images = torch.from_numpy(numpy.load(images_file))
            masks = torch.from_numpy(numpy.load(masks_file))

            for name, tensor in (('images', images), ('masks', masks)):
                if tensor.ndimension() != 5:
                    raise ValueError(f'expected 5D {name}, '
                                     f'got {tensor.ndimension()}D '
                                     f'in layer {layer}')
            if images.shape[:2] != masks.shape[:2]:
                raise ValueError(f'layer {layer} masks/images have '
                                 'different # unit/images: '
                                 f'{images.shape[:2]} vs. {masks.shape[:2]}')
            if images.shape[3:] != masks.shape[3:]:
                raise ValueError(f'layer {layer} masks/images have '
                                 'different height/width '
                                 f'{images.shape[3:]} vs. {masks.shape[3:]}')

            images = images.float()
            masks = masks.float()

            shape = images.shape
            images = images.view(-1, *shape[2:])
            images = renormalizer(images)
            images = images.view(*shape)

            if device is not None:
                images = images.to(device)
                masks = masks.to(device)

            self.images_by_layer[layer] = images
            self.masks_by_layer[layer] = masks

        self.samples = []
        for layer in layers:
            units = zip(self.images_by_layer[layer],
                        self.masks_by_layer[layer])
            for unit, (images, masks) in enumerate(units):
                sample = TopImages(layer=str(layer),
                                   unit=unit,
                                   images=images,
                                   masks=masks)
                self.samples.append(sample)

    def __getitem__(self, index: int) -> TopImages:
        """Return the top images.

        Args:
            index (int): Sample index.

        Returns:
            TopImages: The sample.

        """
        return self.samples[index]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def lookup(self, layer: Layer, unit: int) -> TopImages:
        """Lookup top images for given layer and unit.

        Args:
            layer (Layer): The layer name.
            unit (int): The unit number.

        Raises:
            KeyError: If no top images for given layer and unit.

        Returns:
            TopImages: The top images.

        """
        layer = str(layer)
        if layer not in self.images_by_layer:
            raise KeyError(f'layer "{layer}" does not exist')
        if unit >= len(self.images_by_layer[layer]):
            raise KeyError(f'layer "{layer}" has no unit {unit}')
        return TopImages(layer=layer,
                         unit=unit,
                         images=self.images_by_layer[layer][unit],
                         masks=self.masks_by_layer[layer][unit])

    def unit(self, index: int) -> Unit:
        """Return the unit at the given index.

        Args:
            index (int): Sample index.

        Returns:
            Unit: Layer and unit number.

        """
        sample = self[index]
        return sample.layer, sample.unit

    def units(self, indices: Sequence[int]) -> Sequence[Unit]:
        """Return the units at the given indices.

        Args:
            indices (Sequence[int]): Sample indices.

        Returns:
            Sequence[Unit]: Layer and unit numbers.

        """
        units = [self.unit(index) for index in indices]
        return tuple(units)

    @property
    def k(self) -> int:
        """Return the "k" in "top-k images"."""
        assert len(self) > 0, 'empty dataset?'
        return self.samples[0].images.shape[0]


DEFAULT_LAYER_COLUMN = 'layer'
DEFAULT_UNIT_COLUMN = 'unit'
DEFAULT_ANNOTATION_COLUMN = 'summary'
DEFAULT_ANNOTATIONS_FILE_NAME = 'annotations.csv'


class AnnotatedTopImages(NamedTuple):
    """Top images and annotation for a unit."""

    layer: str
    unit: int
    images: torch.Tensor
    masks: torch.Tensor
    annotations: Sequence[str]

    def as_pil_image_grid(self, **kwargs: Any) -> Image.Image:
        """Show masked top images as a PIL image grid.

        Keyword arguments are forwarded to `TopImages.as_pil_image_grid`.
        """
        return TopImages(*self[:-1]).as_pil_image_grid(**kwargs)


class AnnotatedTopImagesDataset(data.Dataset):
    """Same as TopImagesDataset, but each unit also has annotations."""

    def __init__(self,
                 root: PathLike,
                 *args: Any,
                 annotations_csv_file: Optional[PathLike] = None,
                 layer_column: str = DEFAULT_LAYER_COLUMN,
                 unit_column: str = DEFAULT_UNIT_COLUMN,
                 annotation_column: str = DEFAULT_ANNOTATION_COLUMN,
                 keep_unannotated_samples: bool = False,
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
            keep_unannotated_samples (bool, optional): Keep top images for
                units with no annotations. Otherwise, do not include them
                in the dataset. Defaults to False.

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
            layer_str = row[layer_column]
            layer: Layer = int(layer_str) if layer_str.isdigit() else layer_str
            unit = int(row[unit_column])
            annotation = row[annotation_column]
            annotations[layer, unit].append(annotation)

        samples = []
        top_images_dataset = TopImagesDataset(root, *args, **kwargs)
        if keep_unannotated_samples:
            for top_images in top_images_dataset.samples:
                layer, unit = top_images.layer, top_images.unit
                annotated_top_images = AnnotatedTopImages(
                    layer=top_images.layer,
                    unit=top_images.unit,
                    images=top_images.images,
                    masks=top_images.masks,
                    annotations=tuple(annotations[layer, unit]))
                samples.append(annotated_top_images)
        else:
            for layer, unit in annotations.keys():
                top_images = top_images_dataset.lookup(layer, unit)
                annotated_top_images = AnnotatedTopImages(
                    layer=top_images.layer,
                    unit=top_images.unit,
                    images=top_images.images,
                    masks=top_images.masks,
                    annotations=tuple(annotations[layer, unit]))
                samples.append(annotated_top_images)
        self.samples = tuple(samples)
        self.samples_by_layer_unit = {(s.layer, s.unit): s for s in samples}

        self.name = top_images_dataset.name

    def __getitem__(self, index: int) -> AnnotatedTopImages:
        """Return the annotated top images.

        Args:
            index (int): Sample index.

        Returns:
            AnnotatedTopImages: The sample.

        """
        return self.samples[index]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def lookup(self, layer: Layer, unit: int) -> AnnotatedTopImages:
        """Lookup annotated top images for given layer and unit.

        Args:
            layer (Layer): The layer name.
            unit (int): The unit number.

        Raises:
            KeyError: If no top images for given layer and unit.

        Returns:
            AnnotatedTopImages: The annotated top images.

        """
        key = (str(layer), unit)
        if key not in self.samples_by_layer_unit:
            raise KeyError(f'no annotated top images for: {key}')
        sample = self.samples_by_layer_unit[key]
        return sample

    def unit(self, index: int) -> Unit:
        """Return the unit at the given index.

        Args:
            index (int): Sample index.

        Returns:
            Unit: Layer and unit number.

        """
        sample = self[index]
        return sample.layer, sample.unit

    def units(self, indices: Sequence[int]) -> Sequence[Unit]:
        """Return the units at the given indices.

        Args:
            indices (Sequence[int]): Sample indices.

        Returns:
            Sequence[Unit]: Layer and unit numbers.

        """
        units = [self.unit(index) for index in indices]
        return tuple(units)

    @property
    def k(self) -> int:
        """Return the "k" in "top-k images"."""
        assert len(self) > 0, 'empty dataset?'
        return self.samples[0].images.shape[0]
