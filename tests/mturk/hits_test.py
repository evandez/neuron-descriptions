"""Unit tests for the vocabulary/mturk/hits module."""
import csv
import pathlib
import tempfile

from tests import conftest
from lv import datasets
from lv.mturk import hits

import pytest


@pytest.fixture
def top_images_dataset(top_images_root):
    """Return a fake TopImagesDataset for testing."""
    return datasets.TopImagesDataset(top_images_root)


@pytest.yield_fixture
def csv_file():
    """Return path to output CSV file."""
    with tempfile.TemporaryDirectory() as tempdir:
        yield pathlib.Path(tempdir) / 'hits.csv'


def generate_urls(layer, unit):
    """Generate fake URLs for the given layer and unit."""
    return [
        f'https://images.com/{layer}/{unit}/im-{index}.png'
        for index in range(conftest.N_TOP_IMAGES_PER_UNIT)
    ]


def test_generate_hits_csv(top_images_dataset, csv_file):
    """Test generate_hits_csv constructs expected CSV format."""
    hits.generate_hits_csv(top_images_dataset,
                           csv_file,
                           generate_urls,
                           display_progress=False,
                           validate_urls=False)

    with csv_file.open('r') as handle:
        actual = list(csv.reader(handle))

    header = ['layer', 'unit']
    header += [
        f'image_url_{index + 1}'
        for index in range(conftest.N_TOP_IMAGES_PER_UNIT)
    ]

    lus = [(conftest.layer(layer), conftest.unit(unit))
           for layer in range(conftest.N_LAYERS)
           for unit in range(conftest.N_UNITS_PER_LAYER)]
    rows = [[layer, unit] + generate_urls(layer, unit) for layer, unit in lus]

    expected = [header] + rows
    assert actual == expected


def test_generate_hits_csv_too_few_urls(top_images_dataset, csv_file):
    """Test generate_hits_csv handles when fewer URLs are returned."""
    urls = ['a', 'b']
    hits.generate_hits_csv(top_images_dataset,
                           csv_file,
                           lambda *_: urls,
                           display_progress=False,
                           validate_urls=False)

    with csv_file.open('r') as handle:
        actual = list(csv.reader(handle))

    header = ['layer', 'unit']
    header += [
        f'image_url_{index + 1}'
        for index in range(conftest.N_TOP_IMAGES_PER_UNIT)
    ]

    lus = [(conftest.layer(layer), conftest.unit(unit))
           for layer in range(conftest.N_LAYERS)
           for unit in range(conftest.N_UNITS_PER_LAYER)]
    rows = [[layer, unit] + urls + [''] * (conftest.N_TOP_IMAGES_PER_UNIT - 2)
            for layer, unit in lus]

    expected = [header] + rows
    assert actual == expected


def test_generate_hits_csv_too_many_urls(top_images_dataset, csv_file):
    """Test generate_hits_csv dies when given too many URLs."""
    with pytest.raises(ValueError, match='.*generate_urls.*'):
        hits.generate_hits_csv(top_images_dataset,
                               csv_file,
                               lambda *_: ['bad.com'] *
                               (conftest.N_TOP_IMAGES_PER_UNIT + 1),
                               display_progress=False,
                               validate_urls=False)
