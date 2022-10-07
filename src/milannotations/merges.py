"""Tools for merging (masks, descriptions) with source images.

Because we cannot redistribute ImageNet, we must merge the data that we *can*
distribute (including top-activation masks, human descriptions, image IDs) with
source images obtained by you, the client.

For these tools to work, you must have ImageNet downloaded and stored under
$MILAN_DATA_DIR (defaulting to ./data). The filter structure should look like:

    $MILAN_DATA_DIR/
        imagenet/
            val/
                ...

This is achievable by just unzipping ImageNet in its default format in the data
directory.
"""
import csv
import pathlib
from typing import Any, Optional, Sized, cast

from src.deps.netdissect import renormalize
from src.milannotations import datasets
from src.utils import env
from src.utils.typing import PathLike

import numpy
import torch
import torchvision.datasets
from torch.utils import data
from torchvision import transforms
from tqdm.auto import tqdm


def merge(root: PathLike,
          source: data.Dataset,
          force: bool = False,
          image_index: int = 0) -> None:
    """Merge the source dataset with our data.

    For each layer, this file reads the list of image IDs (in ids.csv), reads
    the corresponding image from the source dataset, and packages the full
    set of source images into a single .npy file for efficiency. Those files
    are then written to root.

    The source dataset must be formatted correctly. The images should all
    be torch tensors, must have the same shape with channels first, and
    have values in [0, 1]. This is all easily accomplished with torchvision
    transforms, for example:

        from torchvision import datasets, transforms
        dataset = datasets.ImageFolder(
            'path/to/imagenet',
            transform=transforms.Compose(
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ),
        )

    Args:
        root (PathLike): Root directory for the MILANNOTATIONS data.
        source (data.Dataset): The source dataset.
        force (bool, optional): If set, overwrite existing merged images.
            Defaults to False.
        image_index (int, optional): Index of the image in each sample.
            Defaults to 0.

    Raises:
        FileNotFoundError: If the IDs CSV is missing.
        IndexError: If any source ID is not valid for the given source dataset.
        ValueError: If the images are malformed in any way.

    """
    root = pathlib.Path(root)
    source_length = len(cast(Sized, source))
    layers = [path for path in root.iterdir() if path.is_dir()]
    renormalizer = renormalize.renormalizer(source='pt', target='byte')

    message = 'merging source images'
    progress = tqdm(layers, desc=message)
    for layer_dir in progress:
        layer = layer_dir.name
        progress.set_description(f'{message} (layer {layer})')

        images_file = layer_dir / 'images.npy'
        if images_file.exists() and not force:
            continue

        ids_file = layer_dir / 'ids.csv'
        if not ids_file.is_file():
            raise FileNotFoundError(f'layer {layer} missing ids.csv')

        with ids_file.open('r') as handle:
            ids_by_unit = tuple(csv.reader(handle))

        images_by_unit = []
        for unit, ids in enumerate(ids_by_unit):
            images = []
            for pos, idx_str in enumerate(ids):
                # It's our code that produces the IDs file, so this should
                # always be valid...
                assert idx_str.isdigit(), idx_str
                idx = int(idx_str)

                # Check the ID is valid.
                if idx < 0 or idx >= source_length:
                    raise IndexError(
                        f'while merging source image {pos} for unit {unit} '
                        f'in layer {layer}, found source ID {idx} which is '
                        f'not valid for source of size {source_length})')

                # Check the image is in fact a tensor.
                image = source[idx][image_index]
                if not isinstance(image, torch.Tensor):
                    raise ValueError(
                        f'while merging source image {pos} for unit {unit} '
                        f'in layer {layer}, found source image of type '
                        f'{type(image).__name__}; it should be a torch tensor')

                images.append(image)

            # Check all images have the same size.
            shapes = {image.shape for image in images}
            if len(shapes) != 1:
                raise ValueError(
                    f'while merging source images for unit {unit} '
                    f'in layer {layer}, found source images with different '
                    f'sizes: {shapes}')

            # Check the one true size also matches expectations.
            shape, = tuple(shapes)
            if len(shape) != 3:
                raise ValueError(
                    f'while merging source images for unit {unit} in layer '
                    f'{layer}, found unexpected image shape; '
                    'source images should be 3D and formatted like '
                    f'(channels, height, width), but got {shape}')
            if shape[0] != 3:
                raise ValueError(
                    f'while merging source images for unit {unit} '
                    f'in layer {layer}, found unexpected image shape; '
                    'source images should be channels-first, with 3 channels, '
                    f'but got shape {shape}')

            # Sanity check images are in correct range.
            images_stacked = torch.stack(images)
            if images_stacked.min().lt(0.) or images_stacked.max().gt(1.):
                raise ValueError(
                    f'while merging source images for unit {unit} '
                    f'in layer {layer}, found pixel with value not in [0, 1]; '
                    'did you forget to normalize?')

            # We're good! Throw em in.
            images_by_unit.append(renormalizer(images_stacked).byte())

        numpy.save(f'{layer_dir}/images.npy', torch.stack(images_by_unit))


def maybe_merge_and_load_dataset(
        root: PathLike,
        source: Optional[str] = None,
        annotations: bool = True,
        force: bool = False,
        image_index: int = 0,
        **kwargs: Any) -> datasets.AnyTopImagesDataset:
    """Load the top images dataset, merging source images if necessary.

    Args:
        root (PathLike): Root directory for the dataset.
        source (str, optional): Name of the source dataset.
            This will be read from $MILAN_DATA_DIR using an ImageFolder.
            Defaults to None.
        annotations (bool, optional): If set, use load annotations with final
            dataset when possible. Otherwise just load top images.
            Defaults to True.
        force (bool, optional): Passed to merge, see above.
            Defaults to False.
        image_index (int, optional): Passed to merge, see above.
            Defaults to 0.

    Returns:
        datasets.AnyTopImagesDataset: The loaded dataset, post merging.

    """
    root = pathlib.Path(root)
    layer_dirs = [path for path in root.iterdir() if path.is_dir()]

    needs_merge = False
    for layer_dir in layer_dirs:
        images_file = layer_dir / 'images.npy'
        if not images_file.exists():
            needs_merge = True

    if needs_merge:
        if source is None:
            raise ValueError('>= 1 layers are missing missing source images '
                             'and no source dataset was provided')
        eg_masks_file = root / next(root.iterdir()) / 'masks.npy'
        if not eg_masks_file.exists():
            raise FileNotFoundError(
                f'tried to find example masks from {eg_masks_file} '
                'but it does not exist?')
        eg_masks = numpy.load(eg_masks_file)
        source_shape = eg_masks.shape[-2:]

        source_dir = env.data_dir() / source
        if not source_dir.exists():
            key = f'{root.parent.name}/{root.name}'
            raise FileNotFoundError(
                f'milannotations "{key}" is not packaged with source images; '
                f'you need to download the source dataset ({source}) '
                'and store in under $MILAN_DATA_DIR, which defaults '
                'to ./data')
        source_dataset = torchvision.datasets.ImageFolder(
            str(source_dir),
            transform=transforms.Compose([
                transforms.Resize(source_shape),
                transforms.ToTensor(),
            ]))
        merge(root, source_dataset, force=force, image_index=image_index)

    annotations_file = root / 'annotations.csv'
    has_annotations = annotations_file.exists()
    if annotations and has_annotations:
        return datasets.AnnotatedTopImagesDataset(root, **kwargs)
    else:
        return datasets.TopImagesDataset(root, **kwargs)
