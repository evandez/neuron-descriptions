"""Unit tests for the `src.exemplars.compute` module."""
import collections
import csv
import math
import pathlib
import tempfile

from src.exemplars import compute
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
        compute.compute(lambda *_: None, lambda *_: None, dataset, **kwargs)


def assert_results_dir_populated(results_dir, layer=None, units=None):
    """Assert the results_dir contains exemplars."""
    if layer is not None:
        results_dir = results_dir / layer
    assert results_dir.is_dir()

    # Check units.
    if units is not None:
        n_units = len(units)
        units_file = results_dir / 'units.npy'
        assert units_file.is_file()
        actual_units = numpy.load(units_file)
        assert actual_units.shape == (n_units,)
        assert tuple(actual_units) == tuple(sorted(units))
    else:
        n_units = conftest.N_UNITS_PER_LAYER

    # Check top images.
    images_file = results_dir / 'images.npy'
    assert images_file.is_file()
    images = numpy.load(images_file)
    assert images.shape == (n_units, conftest.N_TOP_IMAGES_PER_UNIT,
                            *conftest.IMAGE_SHAPE)
    assert images.dtype == numpy.uint8
    assert images.min() >= 0
    assert images.max() <= 255

    # Check top image masks.
    masks_file = results_dir / 'masks.npy'
    assert masks_file.is_file()
    masks = numpy.load(masks_file)
    assert masks.shape == (n_units, conftest.N_TOP_IMAGES_PER_UNIT,
                           *conftest.MASK_SHAPE)
    assert masks.dtype == numpy.uint8
    assert masks.min() >= 0
    assert masks.max() <= 1

    # Check top image IDs.
    ids_file = results_dir / 'ids.csv'
    assert ids_file.is_file()
    with ids_file.open('r') as handle:
        ids = tuple(csv.reader(handle))
    assert len(ids) == n_units
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
    assert len(activations) == n_units
    for unit_activations in activations:
        assert len(unit_activations) == conftest.N_TOP_IMAGES_PER_UNIT
        for unit_activation in unit_activations:
            unit_activation = float(unit_activation)
            assert not math.isnan(float(unit_activation))


def assert_viz_dir_populated(viz_dir, layer=None, units=None):
    """Assert viz_dir contains individual png images and lightbox."""
    if layer is not None:
        viz_dir = viz_dir / layer
    assert viz_dir.is_dir()

    assert viz_dir.is_dir()
    units = range(conftest.N_UNITS_PER_LAYER) if units is None else units
    unit_keys = tuple(f'unit_{unit}' for unit in sorted(units))
    actual = tuple(path.name for path in viz_dir.iterdir())
    assert sorted(actual) == sorted(unit_keys)
    for unit_key in unit_keys:
        viz_unit_dir = viz_dir / unit_key
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
    """Return a fake torch model to compute exemplars for."""
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


@pytest.mark.parametrize(
    'save_results,save_viz',
    (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ),
)
def test_discriminative_no_layer(model, dataset, results_dir, viz_dir,
                                 tally_cache_file, masks_cache_file,
                                 save_results, save_viz):
    """Test discriminative runs when layer not set."""
    compute.discriminative(model,
                           dataset,
                           device='cpu',
                           results_dir=results_dir,
                           viz_dir=viz_dir,
                           display_progress=False,
                           num_workers=1,
                           k=conftest.N_TOP_IMAGES_PER_UNIT,
                           image_size=conftest.IMAGE_SIZE,
                           output_size=conftest.IMAGE_SIZE,
                           save_results=save_results,
                           save_viz=save_viz,
                           tally_cache_file=tally_cache_file,
                           masks_cache_file=masks_cache_file,
                           clear_cache_files=True,
                           clear_results_dir=True)
    if save_results:
        assert_results_dir_populated(results_dir, layer='outputs')
    if save_viz:
        assert_viz_dir_populated(viz_dir, layer='outputs')


@pytest.mark.parametrize(
    'save_results,save_viz',
    (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ),
)
def test_discriminative_layer(model, dataset, results_dir, viz_dir,
                              tally_cache_file, masks_cache_file, save_results,
                              save_viz):
    """Test discriminative runs when layer is set."""
    compute.discriminative(model,
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
                           save_results=save_results,
                           save_viz=save_viz,
                           tally_cache_file=tally_cache_file,
                           masks_cache_file=masks_cache_file,
                           clear_cache_files=True,
                           clear_results_dir=True)
    if save_results:
        assert_results_dir_populated(results_dir, layer='conv_2')
    if save_viz:
        assert_viz_dir_populated(viz_dir, layer='conv_2')


UNITS = (0, 1)


def test_discriminative_units(model, dataset, results_dir, viz_dir,
                              tally_cache_file, masks_cache_file):
    """Test discriminative runs when subset of units specified."""
    compute.discriminative(model,
                           dataset,
                           layer='conv_2',
                           device='cpu',
                           results_dir=results_dir,
                           viz_dir=viz_dir,
                           display_progress=False,
                           num_workers=1,
                           k=conftest.N_TOP_IMAGES_PER_UNIT,
                           units=UNITS,
                           image_size=conftest.IMAGE_SIZE,
                           output_size=conftest.IMAGE_SIZE,
                           tally_cache_file=tally_cache_file,
                           masks_cache_file=masks_cache_file,
                           clear_cache_files=True,
                           clear_results_dir=True)
    assert_results_dir_populated(results_dir, layer='conv_2', units=UNITS)
    assert_viz_dir_populated(viz_dir, layer='conv_2', units=UNITS)


class FeaturesToImage(nn.Module):
    """A dummy module that converts conv features to fake images."""

    def forward(self, features):
        """Convert features to images."""
        assert features.ndimension() == 4
        assert features.shape[1] >= 3
        return torch.sigmoid(features[:, :3])


@pytest.mark.parametrize(
    'save_results,save_viz',
    (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ),
)
def test_generative(model, dataset, results_dir, viz_dir, tally_cache_file,
                    masks_cache_file, save_results, save_viz):
    """Test generative runs in normal case."""
    layers = list(model.named_children())
    layers.append(('output', FeaturesToImage()))
    model = nn.Sequential(collections.OrderedDict(layers))
    compute.generative(model,
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
                       save_results=save_results,
                       save_viz=save_viz,
                       tally_cache_file=tally_cache_file,
                       masks_cache_file=masks_cache_file,
                       clear_cache_files=True,
                       clear_results_dir=True)
    if save_results:
        assert_results_dir_populated(results_dir, layer='conv_2')
    if save_viz:
        assert_viz_dir_populated(viz_dir, layer='conv_2')


def test_generative_units(model, dataset, results_dir, viz_dir,
                          tally_cache_file, masks_cache_file):
    """Test generative runs when subset of units specified."""
    layers = list(model.named_children())
    layers.append(('output', FeaturesToImage()))
    model = nn.Sequential(collections.OrderedDict(layers))
    compute.generative(model,
                       dataset,
                       'conv_2',
                       device='cpu',
                       results_dir=results_dir,
                       viz_dir=viz_dir,
                       display_progress=False,
                       num_workers=1,
                       k=conftest.N_TOP_IMAGES_PER_UNIT,
                       units=UNITS,
                       image_size=conftest.IMAGE_SIZE,
                       output_size=conftest.IMAGE_SIZE,
                       tally_cache_file=tally_cache_file,
                       masks_cache_file=masks_cache_file,
                       clear_cache_files=True,
                       clear_results_dir=True)
    assert_results_dir_populated(results_dir, layer='conv_2', units=UNITS)
    assert_viz_dir_populated(viz_dir, layer='conv_2', units=UNITS)
