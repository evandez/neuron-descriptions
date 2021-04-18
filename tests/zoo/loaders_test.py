"""Unit tests for lv/zoo/loaders module."""
import pathlib
import tempfile

from lv.zoo import loaders

import torch
from torch import nn
from torch.utils import data
import pytest

LAYER = 'my-layer'


class FakeModelConfig:
    """A fake model config object."""

    def __init__(self, model, *args, **kwargs):
        """Initialize the config."""
        self.model = model
        self.args = args
        self.kwargs = kwargs

    def load(self, *args, **kwargs):
        """Fake load that just assert args and kwargs are correct."""
        assert args == self.args
        assert kwargs == self.kwargs
        return self.model, [LAYER]


class DangerConfig:
    """Dangerous config that EXPLODES."""

    def load(self, *_args, **_kwargs):
        """Explode."""
        assert False


@pytest.fixture
def model():
    """Return a fake model for testing."""
    return nn.Sequential(nn.Linear(10, 10))


@pytest.yield_fixture
def model_path(model):
    """Yield a path to a fake model."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'model.pth'
        torch.save(model.state_dict(), path)
        yield path


MODEL = 'my-model'
DATASET = 'my-dataset'


def test_model(model, model_path):
    """Test model correctly reads config."""
    expected = FakeModelConfig(model, model_path, flag=True)
    source = {
        MODEL: {
            DATASET: expected,
            'bad': DangerConfig()
        },
        'bad': {
            'bad': DangerConfig(),
        }
    }
    actuals = loaders.model(MODEL,
                            DATASET,
                            path=model_path,
                            source=source,
                            flag=True)

    assert len(actuals) == 3

    actual_model, actual_layers, actual_config = actuals
    assert actual_model is model
    assert actual_layers == [LAYER]
    assert actual_config is expected


def test_model_no_path(model, model_path):
    """Test model correctly reads config."""
    expected = FakeModelConfig(model,
                               pathlib.Path(__file__).parents[2] /
                               f'.zoo/models/{MODEL}-{DATASET}.pth',
                               flag=True)
    source = {
        MODEL: {
            DATASET: expected,
            'bad': DangerConfig()
        },
        'bad': {
            'bad': DangerConfig(),
        }
    }
    actuals = loaders.model(MODEL, DATASET, source=source, flag=True)

    assert len(actuals) == 3

    actual_model, actual_layers, actual_config = actuals
    assert actual_model is model
    assert actual_layers == [LAYER]
    assert actual_config is expected


@pytest.mark.parametrize('model_key,dataset_key', (
    (MODEL, 'bad'),
    ('bad', DATASET),
    ('bad', 'bad'),
))
def test_model_bad_keys(model, model_key, dataset_key):
    """Test model dies on bad keys."""
    source = {
        MODEL: {
            DATASET: FakeModelConfig(model),
        },
    }
    with pytest.raises(KeyError, match='.*bad.*'):
        loaders.model(model_key, dataset_key, source=source)


class FakeDatasetConfig:
    """A fake dataset config object."""

    def __init__(self, dataset, *args, **kwargs):
        """Initialize the dataset config."""
        self.dataset = dataset
        self.args = args
        self.kwargs = kwargs

    def load(self, *args, **kwargs):
        """Fake load that just assert args and kwargs are correct."""
        assert args == self.args
        assert kwargs == self.kwargs
        return self.dataset


@pytest.fixture
def tensors():
    """Return fake tensor data."""
    return torch.rand(10, 3, 32, 32)


@pytest.fixture
def dataset(tensors):
    """Return a fake dataset."""
    return data.TensorDataset(tensors)


@pytest.yield_fixture
def dataset_path(tensors):
    """Yield a fake dataset path."""
    with tempfile.TemporaryDirectory() as tempdir:
        path = pathlib.Path(tempdir) / 'dataset'
        path.mkdir(exist_ok=True, parents=True)
        torch.save(tensors, path / 'data.pth')
        yield path


def test_dataset(dataset, dataset_path):
    """Test dataset correctly reads config."""
    expected = FakeDatasetConfig(dataset, dataset_path, flag=True)
    source = {DATASET: expected, 'bad': DangerConfig()}
    actual = loaders.dataset(DATASET,
                             path=dataset_path,
                             source=source,
                             flag=True)
    assert actual is dataset


def test_dataset_no_path(dataset):
    """Test dataset correctly reads config."""
    expected = FakeDatasetConfig(dataset,
                                 pathlib.Path(__file__).parents[2] /
                                 f'.zoo/datasets/{DATASET}',
                                 flag=True)
    source = {DATASET: expected, 'bad': DangerConfig()}
    actual = loaders.dataset(DATASET, source=source, flag=True)
    assert actual is dataset


def test_dataset_bad_key(dataset):
    """Test dataset dies on bad keys."""
    source = {
        MODEL: {
            DATASET: FakeDatasetConfig(dataset),
        },
    }
    with pytest.raises(KeyError, match='.*bad.*'):
        loaders.dataset('bad', source=source)
