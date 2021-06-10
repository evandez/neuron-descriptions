"""Unit tests for the lv/dissection/dissect module."""
import collections
import csv
import math
import pathlib
import tempfile

from lv.dissection import dissect
from tests import conftest

import numpy
import pytest
import torch
from torch import nn


@pytest.mark.parametrize('kwargs,error_pattern', (
    (dict(k=-1), '.*k >= 1.*'),
    (dict(quantile=-1), '.*quantile in range.*'),
    (dict(quantile=2), '.*quantile in range.*'),
    (dict(), '.*image_size= must be set.*'),
))
def test_run_bad_inputs(dataset, kwargs, error_pattern):
    """Test run dies on various bad inputs."""
    with pytest.raises(ValueError, match=error_pattern):
        dissect.run(lambda *_: None, lambda *_: None, dataset, **kwargs)


def assert_results_dir_populated(results_dir, layer=None):
    """Assert the results_dir contains dissection results."""
    if layer is not None:
        results_dir = results_dir / layer
    assert results_dir.is_dir()

    # Check top images.
    images_file = results_dir / 'images.npy'
    assert images_file.is_file()
    images = numpy.load(images_file)
    assert images.shape == (conftest.N_UNITS_PER_LAYER,
                            conftest.N_TOP_IMAGES_PER_UNIT,
                            *conftest.IMAGE_SHAPE)
    assert images.dtype == numpy.uint8
    assert images.min() >= 0
    assert images.max() <= 255

    # Check top image masks.
    masks_file = results_dir / 'masks.npy'
    assert masks_file.is_file()
    masks = numpy.load(masks_file)
    assert masks.shape == (conftest.N_UNITS_PER_LAYER,
                           conftest.N_TOP_IMAGES_PER_UNIT,
                           *conftest.MASK_SHAPE)
    assert masks.dtype == numpy.uint8
    assert masks.min() >= 0
    assert masks.max() <= 1

    # Check top image IDs.
    ids_file = results_dir / 'ids.csv'
    assert ids_file.is_file()
    with ids_file.open('r') as handle:
        ids = tuple(csv.reader(handle))
    assert len(ids) == conftest.N_UNITS_PER_LAYER
    for unit_image_ids in ids:
        assert len(unit_image_ids) == conftest.N_TOP_IMAGES_PER_UNIT
        for unit_image_id in unit_image_ids:
            assert unit_image_id.isdigit()
            unit_image_id = int(unit_image_id)
            assert unit_image_id >= 0
            assert unit_image_id < conftest.N_IMAGES_IN_DATASET

    # Check activations for top images.
    activations_file = results_dir / 'activations.csv'
    assert activations_file.is_file()
    with activations_file.open('r') as handle:
        activations = tuple(csv.reader(handle))
    assert len(activations) == conftest.N_UNITS_PER_LAYER
    for unit_activations in activations:
        assert len(unit_activations) == conftest.N_TOP_IMAGES_PER_UNIT
        for unit_activation in unit_activations:
            unit_activation = float(unit_activation)
            assert not math.isnan(float(unit_activation))


def assert_viz_dir_populated(viz_dir, layer=None):
    """Assert viz_dir contains individual png images and lightbox."""
    if layer is not None:
        viz_dir = viz_dir / layer
    assert viz_dir.is_dir()

    assert viz_dir.is_dir()
    units = tuple(f'unit_{unit}' for unit in range(conftest.N_UNITS_PER_LAYER))
    actual = tuple(path.name for path in viz_dir.iterdir())
    assert sorted(actual) == sorted(units)
    for unit in units:
        viz_unit_dir = viz_dir / unit
        assert viz_unit_dir.is_dir()
        for index in range(conftest.N_TOP_IMAGES_PER_UNIT):
            viz_unit_top_image_file = viz_unit_dir / f'image_{index}.png'
            assert viz_unit_top_image_file.is_file()

            viz_unit_lightbox_file = viz_unit_dir / '+lightbox.html'
            assert viz_unit_lightbox_file.is_file()


@pytest.yield_fixture
def results_dir():
    """Yield a fake results directory for testing."""
    with tempfile.TemporaryDirectory() as tempdir:
        results_dir = pathlib.Path(tempdir) / 'results'
        results_dir.mkdir(exist_ok=True, parents=True)
        yield results_dir


@pytest.yield_fixture
def viz_dir():
    """Yield a fake viz directory for testing."""
    with tempfile.TemporaryDirectory() as tempdir:
        viz_dir = pathlib.Path(tempdir) / 'viz'
        viz_dir.mkdir(exist_ok=True, parents=True)
        yield viz_dir


@pytest.fixture
def tally_cache_file(results_dir):
    """Return a fake tally cache file for testing."""
    file = results_dir / 'tally.npz'
    file.touch()
    return file


@pytest.fixture
def masks_cache_file(results_dir):
    """Return a fake masks cache file for testing."""
    file = results_dir / 'masks.npz'
    file.touch()
    return file


@pytest.fixture
def model():
    """Return a fake torch model to dissect."""
    layers = [(
        'conv_1',
        nn.Conv2d(3, conftest.N_UNITS_PER_LAYER, 4, padding=2),
    )]
    for index in range(2, conftest.N_LAYERS + 1):
        layer = (
            f'conv_{index}',
            nn.Conv2d(conftest.N_UNITS_PER_LAYER,
                      conftest.N_UNITS_PER_LAYER,
                      4,
                      padding=2),
        )
        layers.append(layer)
    return nn.Sequential(collections.OrderedDict(layers))


def test_discriminative(model, dataset, results_dir, viz_dir, tally_cache_file,
                        masks_cache_file):
    """Test discriminative runs in normal case."""
    dissect.discriminative(model,
                           dataset,
                           device='cpu',
                           results_dir=results_dir,
                           viz_dir=viz_dir,
                           display_progress=False,
                           num_workers=1,
                           k=conftest.N_TOP_IMAGES_PER_UNIT,
                           image_size=conftest.IMAGE_SIZE,
                           output_size=conftest.IMAGE_SIZE,
                           tally_cache_file=tally_cache_file,
                           masks_cache_file=masks_cache_file,
                           clear_cache_files=True,
                           clear_results_dir=True)
    assert_results_dir_populated(results_dir)
    assert_viz_dir_populated(viz_dir)


def test_sequential(model, dataset, results_dir, viz_dir, tally_cache_file,
                    masks_cache_file):
    """Test sequential runs in normal case."""
    dissect.sequential(model,
                       dataset,
                       layer='conv_2',
                       device='cpu',
                       results_dir=results_dir,
                       viz_dir=viz_dir,
                       display_progress=False,
                       num_workers=1,
                       k=conftest.N_TOP_IMAGES_PER_UNIT,
                       image_size=conftest.IMAGE_SIZE,
                       output_size=conftest.IMAGE_SIZE,
                       tally_cache_file=tally_cache_file,
                       masks_cache_file=masks_cache_file,
                       clear_cache_files=True,
                       clear_results_dir=True)
    assert_results_dir_populated(results_dir, layer='conv_2')
    assert_viz_dir_populated(viz_dir, layer='conv_2')


class FeaturesToImage(nn.Module):
    """A dummy module that converts conv features to fake images."""

    def forward(self, features):
        """Convert features to images."""
        assert features.ndimension() == 4
        assert features.shape[1] >= 3
        return torch.sigmoid(features[:, :3])


def test_generative(model, dataset, results_dir, viz_dir, tally_cache_file,
                    masks_cache_file):
    """Test generative runs in normal case."""
    layers = list(model.named_children())
    layers.append(('output', FeaturesToImage()))
    model = nn.Sequential(collections.OrderedDict(layers))
    dissect.generative(model,
                       dataset,
                       'conv_2',
                       device='cpu',
                       results_dir=results_dir,
                       viz_dir=viz_dir,
                       display_progress=False,
                       num_workers=1,
                       k=conftest.N_TOP_IMAGES_PER_UNIT,
                       image_size=conftest.IMAGE_SIZE,
                       output_size=conftest.IMAGE_SIZE,
                       tally_cache_file=tally_cache_file,
                       masks_cache_file=masks_cache_file,
                       clear_cache_files=True,
                       clear_results_dir=True)
    assert_results_dir_populated(results_dir, layer='conv_2')
    assert_viz_dir_populated(viz_dir, layer='conv_2')
