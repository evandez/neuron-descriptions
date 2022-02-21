"""Unit tests for the `src.mturk.hits` module."""
import csv
import pathlib
import tempfile

from tests import conftest
from src.milannotations import datasets
from src.mturk import hits

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


def generate_urls(layer, unit, k):
    """Generate fake URLs for the given layer and unit."""
    return [
        f'https://images.com/{layer}/{unit}/im-{index}.png'
        for index in range(k)
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

    lus = [(conftest.layer(layer), str(unit))
           for layer in range(conftest.N_LAYERS)
           for unit in range(conftest.N_UNITS_PER_LAYER)]
    rows = [[layer, unit] +
            generate_urls(layer, unit, conftest.N_TOP_IMAGES_PER_UNIT)
            for layer, unit in lus]

    expected = [header] + rows
    assert actual == expected


def test_generate_hits_csv_limit(top_images_dataset, csv_file):
    """Test generate_hits_csv can limit number of generated HITS."""
    hits.generate_hits_csv(top_images_dataset,
                           csv_file,
                           generate_urls,
                           limit=2,
                           display_progress=False,
                           validate_urls=False)

    with csv_file.open('r') as handle:
        actuals = list(csv.reader(handle))
    assert len(actuals) == 3
    keys = {tuple(actual[:2]) for actual in actuals[1:]}
    assert len(keys) == 2


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

    lus = [(conftest.layer(layer), str(unit))
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


RESULTS_CSV = f'''\
Input.layer,ignore1,Input.unit,ignore2,Answer.summary,ignore3,RejectionTime
"{conftest.layer(0)}",foo,0,bar,"{conftest.annotation(0, 0)}",baz,
"{conftest.layer(1)}",foo,1,bar,"{conftest.annotation(1, 1)}",baz,
"{conftest.layer(2)}",foo,2,bar,"{conftest.annotation(2, 2)}",baz,
rejected,foo,0,bar,"a rejected annotation",baz,123
'''


@pytest.yield_fixture
def results_csv_file():
    """Yield a fake results csv file for testing."""
    with tempfile.TemporaryDirectory() as tempdir:
        file = pathlib.Path(tempdir) / 'results.csv'
        with file.open('w') as handle:
            handle.write(RESULTS_CSV)
        yield file


@pytest.mark.parametrize('out_csv_file', (None, 'out.csv'))
def test_strip_results_csv(results_csv_file, out_csv_file):
    """Test strip_results_csv correctly strips the CSV."""
    if out_csv_file is not None:
        out_csv_file = results_csv_file.parent / out_csv_file

    hits.strip_results_csv(results_csv_file, out_csv_file=out_csv_file)

    if out_csv_file is None:
        out_csv_file = results_csv_file

    with out_csv_file.open('r') as handle:
        rows = list(csv.reader(handle))

    assert rows == [
        ['layer', 'unit', 'summary'],
        [conftest.layer(0), '0',
         conftest.annotation(0, 0)],
        [conftest.layer(1), '1',
         conftest.annotation(1, 1)],
        [conftest.layer(2), '2',
         conftest.annotation(2, 2)],
    ]


@pytest.mark.parametrize('out_csv_file', (None, 'out.csv'))
def test_strip_results_csv_cleaning(results_csv_file, out_csv_file):
    """Test strip_results_csv correctly cleans annotations."""
    if out_csv_file is not None:
        out_csv_file = results_csv_file.parent / out_csv_file

    hits.strip_results_csv(results_csv_file,
                           out_csv_file=out_csv_file,
                           remove_prefixes=('(', 'l'),
                           remove_substrings=(')',),
                           remove_suffixes=('tation', ' anno'),
                           replace_substrings={'ayer-': ''})

    if out_csv_file is None:
        out_csv_file = results_csv_file

    with out_csv_file.open('r') as handle:
        rows = list(csv.reader(handle))

    assert rows == [
        ['layer', 'unit', 'summary'],
        [conftest.layer(0), '0', '0, 0'],
        [conftest.layer(1), '1', '1, 1'],
        [conftest.layer(2), '2', '2, 2'],
    ]


def test_strip_results_csv_keep_rejected(results_csv_file):
    """Test strip_results_csv keeps rejected HITs when told to do so."""
    out_csv_file = results_csv_file.parent / 'out.csv'
    hits.strip_results_csv(results_csv_file,
                           out_csv_file=out_csv_file,
                           keep_rejected=True)
    with out_csv_file.open('r') as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 5


def test_strip_results_csv_no_results_csv(results_csv_file):
    """Test strip_results_csv dies when results_csv_file does not exist."""
    results_csv_file.unlink()
    with pytest.raises(FileNotFoundError, match=f'.*{results_csv_file}'):
        hits.strip_results_csv(results_csv_file)


@pytest.mark.parametrize('kwargs', (
    {
        'in_layer_column': 'foo'
    },
    {
        'in_unit_column': 'bar'
    },
    {
        'in_annotation_column': 'baz'
    },
))
def test_strip_results_csv_bad_column(results_csv_file, kwargs):
    """Test strip_results_csv dies on missing column."""
    with pytest.raises(KeyError, match='.*missing column.*'):
        hits.strip_results_csv(results_csv_file, **kwargs)
