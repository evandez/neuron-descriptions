"""Unit tests for the vocabulary/datasets module."""
from tests import conftest
from vocabulary import datasets

import pytest
import torch

N_LAYERS = 2
N_UNITS = 5
N_TOP_IMAGES = 10

IMAGE_SIZE = 224
IMAGE_SHAPE = (3, 224, 224)


def test_top_images_dataset_init(top_images_root):
    """Test TopImagesDataset.__init__ sets state correctly."""
    dataset = datasets.TopImagesDataset(top_images_root)
    assert dataset.root is top_images_root
    assert dataset.layers == tuple(f'layer-{i}' for i in range(N_LAYERS))
    assert dataset.transform is datasets.DEFAULT_TRANSFORM
    assert dataset.images is None


@pytest.mark.parametrize('cache', (True, 'cpu', torch.device('cpu')))
def test_top_images_dataset_init_cache(top_images_root, top_image_tensors,
                                       cache):
    """Test TopImagesDataset.__init__ caches correctly."""
    dataset = datasets.TopImagesDataset(top_images_root, cache=cache)
    assert dataset.root is top_images_root
    assert dataset.layers == tuple(f'layer-{i}' for i in range(N_LAYERS))
    assert dataset.transform is datasets.DEFAULT_TRANSFORM

    expected_images = [uis for lis in top_image_tensors for uis in lis]
    assert len(dataset.images) == len(expected_images)

    for actual, expected in zip(dataset.images, expected_images):
        assert actual.allclose(expected, atol=1e-2)


@pytest.mark.parametrize('validate_top_image_counts', (True, False))
def test_top_images_dataset_init_differing_top_image_count(
        top_images_root, validate_top_image_counts):
    """Test TopImagesDataset.__init__ uses validate_top_image_count."""
    file = top_images_root / 'layer-0' / 'unit-0' / 'im-0.png'
    assert file.is_file()
    file.unlink()

    if validate_top_image_counts:
        with pytest.raises(ValueError, match='.*differing.*'):
            datasets.TopImagesDataset(
                top_images_root,
                validate_top_image_counts=validate_top_image_counts)
    else:
        datasets.TopImagesDataset(
            top_images_root,
            validate_top_image_counts=validate_top_image_counts)


@pytest.mark.parametrize('cache', (False, True))
def test_top_images_dataset_getitem(top_images_root, top_image_tensors, cache):
    """Test TopImagesDataset.__getitem__ returns samples in right order."""
    dataset = datasets.TopImagesDataset(top_images_root, cache=cache)
    for layer in range(N_LAYERS):
        for unit in range(N_UNITS):
            index = layer * N_UNITS + unit
            sample = dataset[index]
            assert sample.layer == f'layer-{layer}'
            assert sample.unit == f'unit-{unit}'
            assert sample.images.allclose(top_image_tensors[layer][unit],
                                          atol=1e-2)


def test_top_images_dataset_len(top_images_root):
    """Test TopImagesDataset.__len__ returns correct length."""
    dataset = datasets.TopImagesDataset(top_images_root)
    assert len(dataset) == N_LAYERS * N_UNITS
