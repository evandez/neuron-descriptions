"""Unit tests for the lv/datasets module."""
import csv

from tests import conftest
from lv import datasets

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
    assert dataset.root == top_images_root
    assert dataset.layers == tuple(f'layer-{i}' for i in range(N_LAYERS))
    assert dataset.transform is datasets.DEFAULT_TRANSFORM
    assert dataset.images is None


@pytest.mark.parametrize('cache', (True, 'cpu', torch.device('cpu')))
def test_top_images_dataset_init_cache(top_images_root, top_image_tensors,
                                       cache):
    """Test TopImagesDataset.__init__ caches correctly."""
    dataset = datasets.TopImagesDataset(top_images_root, cache=cache)
    assert dataset.root == top_images_root
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
    rows.append([conftest.layer(0), conftest.unit(0), annotation])
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
                                              top_image_annotations):
    """Test AnnotatedTopImagesDataset.__getitem__ returns right samples."""
    for layer in range(N_LAYERS):
        for unit in range(N_UNITS):
            index = layer * N_UNITS + unit
            sample = annotated_top_images_dataset[index]
            assert sample.layer == f'layer-{layer}'
            assert sample.unit == f'unit-{unit}'
            assert sample.images.allclose(top_image_tensors[layer][unit],
                                          atol=1e-2)
            assert sample.annotations == (top_image_annotations[index][-1],)


def test_annotated_top_images_dataset_len(annotated_top_images_dataset):
    """Test AnnotatedTopImagesDataset.__len__ returns correct length."""
    assert len(annotated_top_images_dataset) == N_LAYERS * N_UNITS
