"""Test data shared across vocabulary modules."""
import csv
import pathlib
import tempfile

import pytest
import torch
from torchvision import transforms

N_LAYERS = 2
N_UNITS_PER_LAYER = 5
N_TOP_IMAGES_PER_UNIT = 10

IMAGE_SIZE = 224
IMAGE_SHAPE = (3, 224, 224)


@pytest.fixture
def top_image_tensors():
    """Return fake images for testing."""
    layer_images = []
    for _ in range(N_LAYERS):
        unit_images = []
        for _ in range(N_UNITS_PER_LAYER):
            images = torch.rand(N_TOP_IMAGES_PER_UNIT, *IMAGE_SHAPE)
            unit_images.append(images)
        layer_images.append(unit_images)
    return layer_images


def layer(index):
    """Return the layer name for the given index."""
    return f'layer-{index}'


def unit(index):
    """Return the unit name for the given index."""
    return f'unit-{index}'


def image(index):
    """Return the image name for the given index."""
    return f'im-{index}.png'


@pytest.yield_fixture
def top_images_root(top_image_tensors):
    """Yield a fake top images root directory for testing."""
    to_pil_image = transforms.ToPILImage()
    with tempfile.TemporaryDirectory() as tempdir:
        root = pathlib.Path(tempdir) / 'root'
        for layer_index, layer_images in enumerate(top_image_tensors):
            layer_dir = root / layer(layer_index)
            for unit_index, unit_images in enumerate(layer_images):
                unit_dir = layer_dir / unit(unit_index)
                unit_dir.mkdir(exist_ok=True, parents=True)
                for unit_image_index, unit_image in enumerate(unit_images):
                    unit_image_file = unit_dir / image(unit_image_index)
                    to_pil_image(unit_image).save(str(unit_image_file))
        yield root


def annotation(layer_index, unit_index):
    """Create a fake annotation for the given layer and unit indices."""
    return f'({layer(layer_index)}, {unit(unit_index)}) annotation'


@pytest.fixture
def top_image_annotations():
    """Return fake annotations for testing."""
    return [[
        layer(layer_index),
        unit(unit_index),
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
