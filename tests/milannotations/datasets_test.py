"""Unit tests for the `src.milannotations.datasets` submodule."""
import csv
import shutil

from tests import conftest
from src.milannotations import datasets

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


@pytest.mark.parametrize('opacity', (0., .5, 1.))
def test_top_images_as_masked_images_tensor(top_images, opacity):
    """Test TopImages.as_masked_image_tensor returns correct shape."""
    actual = top_images.as_masked_images_tensor(opacity=opacity)
    assert actual.shape == (conftest.N_TOP_IMAGES_PER_UNIT,
                            *conftest.IMAGE_SHAPE)


@pytest.mark.parametrize('opacity', (-1, 2))
def test_top_images_as_masked_images_tensor_bad_opacity(top_images, opacity):
    """Test TopImages.as_masked_images_tensor dies on bad opacity."""
    with pytest.raises(ValueError, match=f'.*{opacity}.*'):
        top_images.as_masked_images_tensor(opacity=opacity)


def test_top_images_as_pil_images(top_images):
    """Test TopImages.as_pil_images returns PIL Images."""
    actuals = top_images.as_pil_images()
    for actual in actuals:
        assert isinstance(actual, Image.Image)


@pytest.mark.parametrize('limit', (None, 2))
def test_top_images_as_pil_image_grid(top_images, limit):
    """Test TopImages.as_pil_image_grid returns a PIL Image."""
    actual = top_images.as_pil_image_grid(limit=limit)
    assert isinstance(actual, Image.Image)


@pytest.mark.parametrize('limit', (0, -1))
def test_top_images_as_pil_image_grid_bad_limit(top_images, limit):
    """Test TopImages.as_pil_image_grid dies on bad limit."""
    with pytest.raises(ValueError, match=f'.*{limit}.*'):
        top_images.as_pil_image_grid(limit=limit)


@pytest.mark.parametrize('device', (None, 'cpu', torch.device('cpu')))
def test_top_images_dataset_init(top_images_root, device):
    """Test TopImagesDataset.__init__ eagerly reads data."""
    dataset = datasets.TopImagesDataset(top_images_root,
                                        display_progress=False,
                                        device=device)
    assert dataset.root == top_images_root
    assert str(top_images_root).endswith(dataset.name)
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


def test_top_images_dataset_init_with_units_file(top_images_root):
    """Test TopImagesDataset.__init__ properly reads units file."""
    layer = conftest.layer(0)
    units = range(conftest.N_UNITS_PER_LAYER - 1)
    layer_dir = top_images_root / layer
    units_file = layer_dir / 'units.npy'
    numpy.save(str(units_file), numpy.array(units))

    dataset = datasets.TopImagesDataset(top_images_root,
                                        display_progress=False)
    assert dataset.root == top_images_root
    assert str(top_images_root).endswith(dataset.name)
    assert dataset.layers == tuple(
        f'layer-{i}' for i in range(conftest.N_LAYERS))
    assert len(dataset.samples) == conftest.N_SAMPLES - 1
    for sample in dataset.samples:
        if sample.layer == layer:
            assert sample.unit != conftest.N_UNITS_PER_LAYER - 1

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


@pytest.mark.parametrize('units,error_pattern', (
    (torch.randint(conftest.N_UNITS_PER_LAYER, size=()), '.*0D.*'),
    (torch.randint(conftest.N_UNITS_PER_LAYER, size=(1, 2)), '.*2D.*'),
))
def test_top_images_dataset_init_bad_units(top_images_root, units,
                                           error_pattern):
    """Test TopImagesDataset.__init__ dies when images/masks misshapen."""
    numpy.save(top_images_root / conftest.layer(0) / 'units.npy', units)
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


@pytest.mark.parametrize('annotation_count', (None, 1))
def test_annotated_top_images_dataset_init_annotation_count(
        top_images_root, top_images_annotations_csv_file,
        top_image_annotations, annotation_count):
    """Test AnnotatedTopImagesDataset.__init__, setting annotation_count."""
    # Remove all L0 annotations.
    banned = conftest.layer(0)
    rows = [conftest.HEADER]
    rows += [anno for anno in top_image_annotations if anno[0] != banned]

    # Add an extra one for L1.
    expanded = conftest.layer(1)
    rows += [anno for anno in top_image_annotations if anno[0] == expanded]

    # Overwrite annotations file with our janky modifications.
    with top_images_annotations_csv_file.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    annotated_top_images_dataset = datasets.AnnotatedTopImagesDataset(
        top_images_root,
        annotations_csv_file=top_images_annotations_csv_file,
        layer_column=conftest.LAYER_COLUMN,
        unit_column=conftest.UNIT_COLUMN,
        annotation_column=conftest.ANNOTATION_COLUMN,
        annotation_count=annotation_count,
        display_progress=False)
    assert str(top_images_root).endswith(annotated_top_images_dataset.name)

    # Yeah, yeah, yeah, this is bad practice, I know...
    if annotation_count is None:
        assert len(annotated_top_images_dataset.samples) == conftest.N_SAMPLES

        actuals = [
            sample for sample in annotated_top_images_dataset.samples
            if sample.layer == banned
        ]
        assert len(actuals) == conftest.N_UNITS_PER_LAYER
        for actual in actuals:
            assert actual.annotations == ()

        actuals = [
            sample for sample in annotated_top_images_dataset.samples
            if sample.layer == expanded
        ]
        assert len(actuals) == conftest.N_UNITS_PER_LAYER
        for actual in actuals:
            assert len(actual.annotations) == 2
    else:
        actual = len(annotated_top_images_dataset.samples)
        expected = (conftest.N_LAYERS - 1) * conftest.N_UNITS_PER_LAYER
        assert actual == expected

        layers = {
            sample.layer for sample in annotated_top_images_dataset.samples
        }
        assert banned not in layers
        assert expanded in layers

        lengths = {
            len(sample.annotations)
            for sample in annotated_top_images_dataset.samples
        }
        assert lengths == {annotation_count}


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
