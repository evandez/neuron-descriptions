"""Test data shared across modules."""
import csv
import pathlib
import tempfile

from lv import datasets

import numpy
import pytest
import torch
from torch.utils import data

N_LAYERS = 2
N_UNITS_PER_LAYER = 3
N_TOP_IMAGES_PER_UNIT = 5
N_SAMPLES = N_LAYERS * N_UNITS_PER_LAYER

IMAGE_SIZE = 16
IMAGE_SHAPE = (3, IMAGE_SIZE, IMAGE_SIZE)
TOP_IMAGES_SHAPE = (N_TOP_IMAGES_PER_UNIT, *IMAGE_SHAPE)

MASK_SHAPE = (1, IMAGE_SIZE, IMAGE_SIZE)
TOP_IMAGES_MASKS_SHAPE = (N_TOP_IMAGES_PER_UNIT, *MASK_SHAPE)

N_IMAGES_IN_DATASET = 10


@pytest.fixture
def dataset():
    """Return a simple image dataset for testing."""
    return data.TensorDataset(torch.rand(N_IMAGES_IN_DATASET, *IMAGE_SHAPE))


@pytest.fixture
def top_image_tensors():
    """Return fake images for testing."""
    layer_images = []
    for _ in range(N_LAYERS):
        unit_images = []
        for _ in range(N_UNITS_PER_LAYER):
            images = torch.randint(256,
                                   size=TOP_IMAGES_SHAPE,
                                   dtype=torch.uint8)
            unit_images.append(images)
        layer_images.append(torch.stack(unit_images))
    return torch.stack(layer_images)


@pytest.fixture
def top_image_masks():
    """Return top image masks for testing."""
    layer_masks = []
    for _ in range(N_LAYERS):
        unit_masks = []
        for _ in range(N_UNITS_PER_LAYER):
            masks = torch.randint(2,
                                  size=TOP_IMAGES_MASKS_SHAPE,
                                  dtype=torch.uint8)
            unit_masks.append(masks)
        layer_masks.append(torch.stack(unit_masks))
    return torch.stack(layer_masks)


def layer(index):
    """Return the layer name for the given index."""
    return f'layer-{index}'


def image(index):
    """Return the image name for the given index."""
    return f'im-{index}.png'


@pytest.yield_fixture
def top_images_root(top_image_tensors, top_image_masks):
    """Yield a fake top images root directory for testing."""
    with tempfile.TemporaryDirectory() as tempdir:
        root = pathlib.Path(tempdir) / 'root'
        layer_data = zip(top_image_tensors, top_image_masks)
        for layer_index, (layer_images, layer_masks) in enumerate(layer_data):
            layer_dir = root / layer(layer_index)
            layer_dir.mkdir(parents=True)
            numpy.save(layer_dir / 'images.npy', layer_images.numpy())
            numpy.save(layer_dir / 'masks.npy', layer_masks.numpy())
        yield root


@pytest.fixture
def top_images_dataset(top_images_root):
    """Return a TopImagesDataset for testing."""
    return datasets.TopImagesDataset(top_images_root)


def annotation(layer_index, unit_index):
    """Create a fake annotation for the given layer and unit indices."""
    return f'({layer(layer_index)}, {unit_index}) annotation'


@pytest.fixture
def top_image_annotations():
    """Return fake annotations for testing."""
    return [[
        layer(layer_index), unit_index,
        annotation(layer_index, unit_index)
    ]
            for layer_index in range(N_LAYERS)
            for unit_index in range(N_UNITS_PER_LAYER)]


LAYER_COLUMN = 'the_layer'
UNIT_COLUMN = 'the_unit'
ANNOTATION_COLUMN = 'the_annotation'


@pytest.fixture
def top_images_annotations_csv_file(top_images_root, top_image_annotations):
    """Return a fake annotations CSV file for testing."""
    rows = [[LAYER_COLUMN, UNIT_COLUMN, ANNOTATION_COLUMN]]
    rows += top_image_annotations

    annotations_csv_file = top_images_root / 'annotations.csv'
    with annotations_csv_file.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    return annotations_csv_file


@pytest.fixture
def annotated_top_images_dataset(top_images_root,
                                 top_images_annotations_csv_file):
    """Return an AnnotatedTopImagesDataset for testing."""
    return datasets.AnnotatedTopImagesDataset(
        top_images_root,
        annotations_csv_file=top_images_annotations_csv_file,
        layer_column=LAYER_COLUMN,
        unit_column=UNIT_COLUMN,
        annotation_column=ANNOTATION_COLUMN)
