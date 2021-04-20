"""Datasets for dissection results and annotations."""
import collections
import csv
import pathlib
from typing import Any, Iterable, NamedTuple, Optional, Sequence, Union

from lv.utils.typing import PathLike
from third_party.netdissect import renormalize

import numpy
import torch
import tqdm
from PIL import Image
from torch.utils import data
from torchvision import utils
from torchvision.transforms import functional


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


class TopImagesDataset(data.Dataset[TopImages]):
    """Top-activating images for invidual units."""

    def __init__(self,
                 root: PathLike,
                 layers: Optional[Iterable[str]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 display_progress: bool = True):
        """Initialize the dataset.

        Args:
            root (PathLike): Root directory for the dataset. See
                `dissection.dissect` function for expected format.
            layers (Optional[Iterable[str]], optional): The layers to load.
                Layer data is assumed to be a subdirectory of the root.
                By default, all subdirectories of root are treated as layers.
            device (Optional[Union[str, torch.device]], optional): Send all
                tensors to this device.
            display_progress (bool, optional): Show the progress bar when
                reading images into menu. Defaults to True.

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
            layers = [str(f.name) for f in root.iterdir() if f.is_dir()]
        if not layers:
            raise ValueError('no layers given and root has no subdirectories')

        self.root = root
        self.layers = layers = tuple(sorted(layers))
        self.device = device

        self.renormalizer = renormalize.renormalizer(source='byte',
                                                     target='pt')

        images_by_layer = {}
        masks_by_layer = {}
        for layer in tqdm.tqdm(layers) if display_progress else layers:
            images_file = root / layer / 'images.npy'
            masks_file = root / layer / 'masks.npy'
            for file in (images_file, masks_file):
                if not file.exists():
                    raise FileNotFoundError(f'{layer} is missing {file.name}')

            images = torch.from_numpy(numpy.load(root / layer / 'images.npy'))
            masks = torch.from_numpy(numpy.load(root / layer / 'masks.npy'))

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
            images = self.renormalizer(images)
            images = images.view(*shape)

            if device is not None:
                images = images.to(device)
                masks = masks.to(device)

            images_by_layer[layer] = images
            masks_by_layer[layer] = masks

        self.samples = []
        for layer in layers:
            units = zip(images_by_layer[layer], masks_by_layer[layer])
            for unit, (images, masks) in enumerate(units):
                sample = TopImages(layer=layer,
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
        sample = self.samples[index]
        if self.device is not None:
            sample = TopImages(layer=sample.layer,
                               unit=sample.unit,
                               images=sample.images.to(self.device),
                               masks=sample.masks.to(self.device))
        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

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

    def as_pil_image_grid(self, **kwargs) -> Image.Image:
        """Show masked top images as a PIL image grid.

        Keyword arguments are forwarded to `TopImages.as_pil_image_grid`.
        """
        return TopImages(*self[:-1]).as_pil_image_grid(**kwargs)


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
            unit = int(row[unit_column])
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
                                  masks=top_images.masks,
                                  annotations=annotations)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.top_images_dataset)
