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
    dataset = datasets.TopImagesDataset(top_images_root, device=device)
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
    dataset = datasets.TopImagesDataset(top_images_root, device='cpu')
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


@pytest.fixture
def annotated_top_images(top_images):
    """Return AnnotatedTopImages for testing."""
    return datasets.AnnotatedTopImages(*top_images, annotations=('foo',))


@pytest.mark.parametrize('opacity', (0, .5, 1))
def test_annotated_top_images_as_pil_image_grid(annotated_top_images, opacity):
    """Test AnnotatedTopImages.as_pil_image_grid returns PIL image."""
    actual = annotated_top_images.as_pil_image_grid(opacity=opacity)
    assert isinstance(actual, Image.Image)


@pytest.mark.parametrize('validate', (False, True))
def test_annotated_top_images_init_validate_top_image_annotated(
        top_images_root, top_images_annotations_csv_file, validate):
    """Test AnnotatedTopImagesDataset.__init__ validates correctly."""
    with top_images_annotations_csv_file.open('r') as handle:
        rows = list(csv.reader(handle))
    with top_images_annotations_csv_file.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerows(rows[:-1])

    if validate:
        with pytest.raises(ValueError, match='validate_top_image_annotated'):
            datasets.AnnotatedTopImagesDataset(
                top_images_root,
                annotations_csv_file=top_images_annotations_csv_file,
                layer_column=conftest.LAYER_COLUMN,
                unit_column=conftest.UNIT_COLUMN,
                annotation_column=conftest.ANNOTATION_COLUMN,
                validate_top_image_annotated=True)
    else:
        dataset = datasets.AnnotatedTopImagesDataset(
            top_images_root,
            annotations_csv_file=top_images_annotations_csv_file,
            layer_column=conftest.LAYER_COLUMN,
            unit_column=conftest.UNIT_COLUMN,
            annotation_column=conftest.ANNOTATION_COLUMN,
            validate_top_image_annotated=False)
        assert dataset[-1].annotations == ()


@pytest.mark.parametrize('validate', (False, True))
def test_annotated_top_images_init_validate_top_image_annotatation_counts(
        top_images_root, top_images_annotations_csv_file, validate):
    """Test AnnotatedTopImagesDataset.__init__ validates correctly."""
    with top_images_annotations_csv_file.open('r') as handle:
        rows = list(csv.reader(handle))
    annotation = 'another annotation!'
    rows.append([conftest.layer(0), 0, annotation])
    with top_images_annotations_csv_file.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    if validate:
        with pytest.raises(ValueError,
                           match='validate_top_image_annotation_counts'):
            datasets.AnnotatedTopImagesDataset(
                top_images_root,
                annotations_csv_file=top_images_annotations_csv_file,
                layer_column=conftest.LAYER_COLUMN,
                unit_column=conftest.UNIT_COLUMN,
                annotation_column=conftest.ANNOTATION_COLUMN,
                validate_top_image_annotation_counts=True)
    else:
        dataset = datasets.AnnotatedTopImagesDataset(
            top_images_root,
            annotations_csv_file=top_images_annotations_csv_file,
            layer_column=conftest.LAYER_COLUMN,
            unit_column=conftest.UNIT_COLUMN,
            annotation_column=conftest.ANNOTATION_COLUMN,
            validate_top_image_annotation_counts=False)
        assert sorted(dataset[0].annotations) == sorted(
            (conftest.annotation(0, 0), annotation))


@pytest.fixture
def annotated_top_images_dataset(top_images_root,
                                 top_images_annotations_csv_file):
    """Return an AnnotatedTopImagesDataset for testing."""
    return datasets.AnnotatedTopImagesDataset(
        top_images_root,
        annotations_csv_file=top_images_annotations_csv_file,
        layer_column=conftest.LAYER_COLUMN,
        unit_column=conftest.UNIT_COLUMN,
        annotation_column=conftest.ANNOTATION_COLUMN)


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
