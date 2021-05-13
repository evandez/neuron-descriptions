"""Unit tests for the lv/datasets module."""
import csv
import shutil

from tests import conftest
from lv import datasets

import numpy
import pytest
import torch
from PIL import Image


@pytest.fixture
def top_images():
    """Return TopImages for testing."""
    return datasets.TopImages(
        layer='layer',
        unit=0,
        images=torch.rand(conftest.N_TOP_IMAGES_PER_UNIT,
                          *conftest.IMAGE_SHAPE),
        masks=torch.randint(2,
                            size=conftest.TOP_IMAGES_MASKS_SHAPE,
                            dtype=torch.float),
    )


@pytest.mark.parametrize('opacity', (0, .5, 1))
def test_top_images_as_pil_image_grid(top_images, opacity):
    """Test TopImages.as_pil_image returns a PIL Image."""
    actual = top_images.as_pil_image_grid(opacity=opacity)
    assert isinstance(actual, Image.Image)


@pytest.mark.parametrize('opacity', (-1, 2))
def test_top_images_as_pil_image_grid_bad_opacity(top_images, opacity):
    """Test TopImages.as_pil_image dies on bad opacity."""
    with pytest.raises(ValueError, match=f'.*{opacity}.*'):
        top_images.as_pil_image_grid(opacity=opacity)


@pytest.mark.parametrize('device', (None, 'cpu', torch.device('cpu')))
def test_top_images_dataset_init(top_images_root, device):
    """Test TopImagesDataset.__init__ eagerly reads data."""
    dataset = datasets.TopImagesDataset(top_images_root,
                                        display_progress=False,
                                        device=device)
    assert dataset.root == top_images_root
    assert dataset.layers == tuple(
        f'layer-{i}' for i in range(conftest.N_LAYERS))
    assert dataset.device is device
    assert len(dataset.samples) == conftest.N_SAMPLES
    for sample in dataset.samples:
        assert sample.images.dtype is torch.float
        assert sample.images.min() >= 0
        assert sample.images.max() <= 1

        assert sample.masks.dtype is torch.float
        assert sample.masks.min() >= 0
        assert sample.masks.max() <= 1


@pytest.mark.parametrize('subpath,error_pattern', (
    ('', '.*root directory not found.*'),
    (f'{conftest.layer(0)}/images.npy', '.*missing images.*'),
    (f'{conftest.layer(0)}/masks.npy', '.*missing masks.*'),
))
def test_top_images_dataset_init_missing_files(top_images_root, subpath,
                                               error_pattern):
    """Test TopImagesDataset.__init__ dies when files are missing."""
    path = top_images_root / subpath
    if path.is_dir():
        shutil.rmtree(path)
    else:
        assert path.is_file()
        path.unlink()

    with pytest.raises(FileNotFoundError, match=error_pattern):
        datasets.TopImagesDataset(top_images_root)


@pytest.mark.parametrize('images,masks,error_pattern', (
    (
        torch.rand(5, 3, 224, 224),
        None,
        '.*5D images.*',
    ),
    (
        None,
        torch.randint(1, size=(5, 1, 224, 224), dtype=torch.uint8),
        '.*5D masks.*',
    ),
    (
        torch.rand(10, 5, 3, 224, 224),
        torch.randint(1, size=(8, 5, 1, 224, 224), dtype=torch.uint8),
        '.*masks/images.*',
    ),
    (
        torch.rand(10, 5, 3, 224, 224),
        torch.randint(1, size=(10, 4, 1, 224, 224), dtype=torch.uint8),
        '.*masks/images.*',
    ),
    (
        torch.rand(10, 5, 3, 223, 224),
        torch.randint(1, size=(10, 5, 1, 224, 224), dtype=torch.uint8),
        '.*height/width.*',
    ),
    (
        torch.rand(10, 5, 3, 224, 223),
        torch.randint(1, size=(10, 5, 1, 224, 224), dtype=torch.uint8),
        '.*height/width.*',
    ),
))
def test_top_images_dataset_init_bad_images_or_masks(top_images_root,
                                                     top_image_tensors,
                                                     top_image_masks, images,
                                                     masks, error_pattern):
    """Test TopImagesDataset.__init__ dies when images/masks misshapen."""
    if images is None:
        images = top_image_tensors[0]
    if masks is None:
        masks = top_image_masks[0]

    for name, tensor in (('images', images), ('masks', masks)):
        numpy.save(top_images_root / conftest.layer(0) / f'{name}.npy', tensor)

    with pytest.raises(ValueError, match=error_pattern):
        datasets.TopImagesDataset(top_images_root)


def test_top_images_dataset_getitem(top_images_root, top_image_tensors,
                                    top_image_masks):
    """Test TopImagesDataset.__getitem__ returns samples in right order."""
    dataset = datasets.TopImagesDataset(top_images_root,
                                        display_progress=False,
                                        device='cpu')
    for layer in range(conftest.N_LAYERS):
        for unit in range(conftest.N_UNITS_PER_LAYER):
            index = layer * conftest.N_UNITS_PER_LAYER + unit
            sample = dataset[index]
            assert sample.layer == f'layer-{layer}'
            assert sample.unit == unit
            assert sample.images.dtype is torch.float
            assert sample.images.allclose(
                top_image_tensors[layer][unit].float() / 255, atol=1e-3)
            assert sample.masks.dtype is torch.float
            assert sample.masks.equal(top_image_masks[layer][unit].float())


def test_top_images_dataset_len(top_images_dataset):
    """Test TopImagesDataset.__len__ returns correct length."""
    assert len(top_images_dataset) == conftest.N_SAMPLES


def test_top_images_dataset_lookup(top_images_dataset, top_image_tensors,
                                   top_image_masks):
    """Test TopImagesDataset.lookup finds correct layer and unit."""
    for layer_index in range(conftest.N_LAYERS):
        layer = conftest.layer(layer_index)
        for unit in range(conftest.N_UNITS_PER_LAYER):
            actual = top_images_dataset.lookup(layer, unit)
            assert actual.layer == layer
            assert actual.unit == unit
            assert actual.images.allclose(
                top_image_tensors[layer_index][unit] / 255, atol=1e-3)
            assert actual.masks.equal(
                top_image_masks[layer_index][unit].float())


@pytest.mark.parametrize('layer,unit,error_pattern', (
    ('layer-10000', 0, '.*"layer-10000" does not exist.*'),
    ('layer-0', 100000, '.*unit 100000.*'),
))
def test_top_images_dataset_lookup_bad_key(top_images_dataset, layer, unit,
                                           error_pattern):
    """Test TopImagesDataset.lookup dies when given a bad key."""
    with pytest.raises(KeyError, match=error_pattern):
        top_images_dataset.lookup(layer, unit)


def test_top_images_dataset_k(top_images_dataset):
    """Test TopImagesDataset.k returns number of top images."""
    assert top_images_dataset.k == conftest.N_TOP_IMAGES_PER_UNIT


@pytest.fixture
def annotated_top_images(top_images):
    """Return AnnotatedTopImages for testing."""
    return datasets.AnnotatedTopImages(*top_images, annotations=('foo',))


@pytest.mark.parametrize('opacity', (0, .5, 1))
def test_annotated_top_images_as_pil_image_grid(annotated_top_images, opacity):
    """Test AnnotatedTopImages.as_pil_image_grid returns PIL image."""
    actual = annotated_top_images.as_pil_image_grid(opacity=opacity)
    assert isinstance(actual, Image.Image)


@pytest.mark.parametrize('keep_unannotated_samples', (False, True))
def test_annotated_top_images_dataset_init_no_keep_unannotated_samples(
        top_images_root, top_images_annotations_csv_file,
        top_image_annotations, keep_unannotated_samples):
    """Test AnnotatedTopImagesDataset.__init__, not keeping unannotated."""
    banned = conftest.layer(0)
    rows = [conftest.HEADER]
    rows += [anno for anno in top_image_annotations if anno[0] != banned]
    with top_images_annotations_csv_file.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    annotated_top_images_dataset = datasets.AnnotatedTopImagesDataset(
        top_images_root,
        annotations_csv_file=top_images_annotations_csv_file,
        layer_column=conftest.LAYER_COLUMN,
        unit_column=conftest.UNIT_COLUMN,
        annotation_column=conftest.ANNOTATION_COLUMN,
        keep_unannotated_samples=keep_unannotated_samples,
        display_progress=False)

    # Yeah, yeah, yeah, this is bad practice, I know...
    if keep_unannotated_samples:
        assert len(annotated_top_images_dataset.samples) == conftest.N_SAMPLES

        actuals = [
            sample for sample in annotated_top_images_dataset.samples
            if sample.layer == banned
        ]
        assert len(actuals) == conftest.N_UNITS_PER_LAYER
        for actual in actuals:
            assert actual.annotations == ()
    else:
        actual = len(annotated_top_images_dataset.samples)
        expected = (conftest.N_LAYERS - 1) * conftest.N_UNITS_PER_LAYER
        assert actual == expected

        layers = {
            sample.layer for sample in annotated_top_images_dataset.samples
        }
        assert banned not in layers


def test_annotated_top_images_dataset_getitem(annotated_top_images_dataset,
                                              top_image_tensors,
                                              top_image_masks,
                                              top_image_annotations):
    """Test AnnotatedTopImagesDataset.__getitem__ returns right samples."""
    for layer in range(conftest.N_LAYERS):
        for unit in range(conftest.N_UNITS_PER_LAYER):
            index = layer * conftest.N_UNITS_PER_LAYER + unit
            sample = annotated_top_images_dataset[index]
            assert sample.layer == conftest.layer(layer)
            assert sample.unit == unit
            assert sample.images.dtype is torch.float
            assert sample.images.allclose(
                top_image_tensors[layer][unit].float() / 255, atol=1e-3)
            assert sample.masks.dtype is torch.float
            assert sample.masks.equal(top_image_masks[layer][unit].float())
            assert sample.annotations == (top_image_annotations[index][-1],)


def test_annotated_top_images_dataset_len(annotated_top_images_dataset):
    """Test AnnotatedTopImagesDataset.__len__ returns correct length."""
    assert len(annotated_top_images_dataset) == conftest.N_SAMPLES


def test_annotated_top_images_dataset_lookup(annotated_top_images_dataset,
                                             top_image_tensors,
                                             top_image_masks,
                                             top_image_annotations):
    """Test AnnotatedTopImagesDataset.lookup finds correct sample."""
    for layer_index in range(conftest.N_LAYERS):
        layer = conftest.layer(layer_index)
        for unit in range(conftest.N_UNITS_PER_LAYER):
            actual = annotated_top_images_dataset.lookup(layer, unit)
            assert actual.layer == layer
            assert actual.unit == unit
            assert actual.images.allclose(
                top_image_tensors[layer_index][unit] / 255, atol=1e-3)
            assert actual.masks.equal(
                top_image_masks[layer_index][unit].float())
            index = layer_index * conftest.N_UNITS_PER_LAYER + unit
            assert actual.annotations == (top_image_annotations[index][-1],)


def test_annotated_top_images_dataset_lookup_bad_key(
        annotated_top_images_dataset):
    """Test AnnotatedTopImagesDataset.lookup dies on bad key."""
    bad = ('layer-10000', 0)
    with pytest.raises(KeyError, match=f'.*{bad}.*'):
        annotated_top_images_dataset.lookup(*bad)


def test_annotated_top_images_dataset_k(annotated_top_images_dataset):
    """Test AnnotatedTopImagesDataset.k returns correct value."""
    assert annotated_top_images_dataset.k == conftest.N_TOP_IMAGES_PER_UNIT
