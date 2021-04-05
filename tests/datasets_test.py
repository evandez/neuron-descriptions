"""Unit tests for the vocabulary/datasets module."""
import pathlib
import tempfile

from vocabulary import datasets

import pytest
import torch
from torchvision import transforms

N_LAYERS = 2
N_UNITS = 5
N_TOP_IMAGES = 10

IMAGE_SIZE = 224
IMAGE_SHAPE = (3, 224, 224)


@pytest.fixture
def images():
    """Return fake images for testing."""
    layer_images = []
    for _ in range(N_LAYERS):
        unit_images = []
        for _ in range(N_UNITS):
            images = torch.rand(N_TOP_IMAGES, *IMAGE_SHAPE)
            unit_images.append(images)
        layer_images.append(unit_images)
    return layer_images


@pytest.yield_fixture
def root(images):
    """Yield a fake top images root directory for testing."""
    to_pil_image = transforms.ToPILImage()
    with tempfile.TemporaryDirectory() as tempdir:
        root = pathlib.Path(tempdir) / 'root'
        for layer_index, layer_images in enumerate(images):
            layer_dir = root / f'layer-{layer_index}'
            for unit_index, unit_images in enumerate(layer_images):
                unit_dir = layer_dir / f'unit-{unit_index}'
                unit_dir.mkdir(exist_ok=True, parents=True)
                for unit_image_index, unit_image in enumerate(unit_images):
                    unit_image_file = unit_dir / f'im-{unit_image_index}.png'
                    to_pil_image(unit_image).save(str(unit_image_file))
        yield root


def test_top_images_dataset_init(root):
    """Test TopImagesDataset.__init__ sets state correctly."""
    dataset = datasets.TopImagesDataset(root)
    assert dataset.root is root
    assert dataset.layers == tuple(f'layer-{i}' for i in range(N_LAYERS))
    assert dataset.transform is datasets.DEFAULT_TRANSFORM
    assert dataset.images is None


@pytest.mark.parametrize('cache', (True, 'cpu', torch.device('cpu')))
def test_top_images_dataset_init_cache(root, images, cache):
    """Test TopImagesDataset.__init__ caches correctly."""
    dataset = datasets.TopImagesDataset(root, cache=cache)
    assert dataset.root is root
    assert dataset.layers == tuple(f'layer-{i}' for i in range(N_LAYERS))
    assert dataset.transform is datasets.DEFAULT_TRANSFORM

    expected_images = [uis for lis in images for uis in lis]
    assert len(dataset.images) == len(expected_images)

    for actual, expected in zip(dataset.images, expected_images):
        assert actual.allclose(expected, atol=1e-2)


@pytest.mark.parametrize('cache', (False, True))
def test_top_images_dataset_getitem(root, images, cache):
    """Test TopImagesDataset.__getitem__ returns samples in right order."""
    dataset = datasets.TopImagesDataset(root, cache=cache)
    for layer in range(N_LAYERS):
        for unit in range(N_UNITS):
            index = layer * N_UNITS + unit
            sample = dataset[index]
            assert sample.layer == f'layer-{layer}'
            assert sample.unit == f'unit-{unit}'
            assert sample.images.allclose(images[layer][unit], atol=1e-2)


def test_top_images_dataset_len(root):
    """Test TopImagesDataset.__len__ returns correct length."""
    dataset = datasets.TopImagesDataset(root)
    assert len(dataset) == N_LAYERS * N_UNITS
