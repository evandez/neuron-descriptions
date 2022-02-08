"""Unit tests for lv/zoo/core module."""
import collections
import pathlib
import tempfile

from src.zoo import core

import pytest
import torch
from torch import nn
from torch.utils import data


class Model(nn.Sequential):
    """A fake model that does not do very much."""

    def __init__(self, *args, flag=False):
        """Initialize the model."""
        super().__init__(
            collections.OrderedDict([('layer', nn.Linear(10, 10))]))
        self.args = args
        self.flag = flag


@pytest.fixture
def weights():
    """Return fake model weights."""
    return Model().state_dict()


@pytest.yield_fixture
def weights_file(weights):
    """Yield a fake weights file."""
    with tempfile.TemporaryDirectory() as tempdir:
        file = pathlib.Path(tempdir) / 'weights.pth'
        torch.save(weights, file)
        yield file


@pytest.fixture
def model_config():
    """Return a ModelConfig for testing."""
    return core.ModelConfig(factory=Model, flag=True)


def test_model_config_load(model_config, weights_file, weights):
    """Test ModelConfig.load in the basic case."""
    model = model_config.load(path=weights_file)

    assert not model.args
    assert model.flag

    state_dict = model.state_dict()
    assert state_dict.keys() == weights.keys()

    for key in state_dict:
        assert state_dict[key].allclose(weights[key], atol=1e-3)


def test_model_config_load_requires_path(weights_file, weights):
    """Test ModelConfig.load when requires_path is set."""
    model_config = core.ModelConfig(factory=Model,
                                    requires_path=True,
                                    flag=True)
    model = model_config.load(path=weights_file)

    assert model.args == (weights_file,)
    assert model.flag

    state_dict = model.state_dict()
    assert state_dict.keys() == weights.keys()

    for key in state_dict:
        assert state_dict[key].allclose(weights[key], atol=1e-3)


def test_model_config_load_overwrite_defaults(model_config, weights_file,
                                              weights):
    """Test ModelConfig.load overwrites defaults."""
    model = model_config.load(path=weights_file, flag=False)

    assert not model.args
    assert not model.flag

    state_dict = model.state_dict()
    assert state_dict.keys() == weights.keys()

    for key in state_dict:
        assert state_dict[key].allclose(weights[key], atol=1e-3)


def test_model_config_load_no_load_weights(model_config, weights_file,
                                           weights):
    """Test ModelConfig.load does not load weights when told not to."""
    model_config.load_weights = False
    model = model_config.load(path=weights_file)

    assert not model.args
    assert model.flag

    state_dict = model.state_dict()
    assert state_dict.keys() == weights.keys()

    for key in state_dict:
        assert not state_dict[key].allclose(weights[key], atol=1e-3)


def test_model_config_load_transform_weights(model_config, weights_file,
                                             weights):
    """Test ModelConfig.load does not load weights when told not to."""
    model_config.transform_weights = lambda state_dict:\
        {key: torch.zeros_like(tensor) for key, tensor in state_dict.items()}
    model = model_config.load(path=weights_file,)

    assert not model.args
    assert model.flag

    state_dict = model.state_dict()
    assert state_dict.keys() == weights.keys()

    for key in state_dict:
        assert state_dict[key].eq(0).all()


def test_model_config_load_bad_weights_path(model_config, weights_file):
    """Test ModelConfig.load dies when given bad weights file."""
    weights_file.unlink()
    with pytest.raises(FileNotFoundError, match='.*model path not found.*'):
        model_config.load(weights_file)


def test_model_config_load_no_path_bad():
    """Test ModelConfig.load dies when it requires path, but no path given."""
    model_config = core.ModelConfig(Model,
                                    requires_path=True,
                                    load_weights=False)
    with pytest.raises(ValueError, match='.*model requires path.*'):
        model_config.load()


class Dataset(data.Dataset):
    """A fake dataset that reads tensors from disk."""

    def __init__(self, path, flag=False):
        """Initialize the dataset."""
        assert path.is_file()
        self.dataset = data.TensorDataset(torch.load(path))
        self.flag = flag

    def __getitem__(self, index):
        """Return the index'th tensor."""
        return self.dataset[index]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataset)


N_SAMPLES = 10
N_FEATURES = 15


@pytest.fixture
def tensors():
    """Return fake tensor data for testing."""
    return torch.rand(N_SAMPLES, N_FEATURES)


@pytest.yield_fixture
def dataset_file(tensors):
    """Return a fake dataset file for testing."""
    with tempfile.TemporaryDirectory() as tempdir:
        file = pathlib.Path(tempdir) / 'data.pth'
        torch.save(tensors, file)
        yield file


@pytest.fixture
def dataset_config():
    """Return a DatasetConfig for testing."""
    return core.DatasetConfig(factory=Dataset, flag=True)


def test_dataset_config_load(dataset_config, dataset_file, tensors):
    """Test DatasetConfig.load correctly instantiates dataset."""
    actual = dataset_config.load(path=dataset_file)
    assert torch.cat(actual.dataset.tensors).allclose(tensors, atol=1e-3)
    assert actual.flag


def test_dataset_config_load_overwrite_defaults(dataset_config, dataset_file,
                                                tensors):
    """Test DatasetConfig.load correctly overwrites defaults."""
    actual = dataset_config.load(path=dataset_file, flag=False)
    assert torch.cat(actual.dataset.tensors).allclose(tensors, atol=1e-3)
    assert not actual.flag


def test_dataset_config_load_no_path_when_not_required(dataset_file, tensors):
    """Test DatasetConfig.load runs when no (optional) path provided."""
    dataset_config = core.DatasetConfig(factory=Dataset,
                                        path=dataset_file,
                                        requires_path=False)
    actual = dataset_config.load()
    assert torch.cat(actual.dataset.tensors).allclose(tensors, atol=1e-3)
    assert not actual.flag


def test_dataset_config_load_no_path_when_required(dataset_config):
    """Test DatasetConfig.load dies when no (required) path provided."""
    with pytest.raises(ValueError, match='.*dataset requires path.*'):
        dataset_config.load()


def test_dataset_config_path_does_not_exist(dataset_config, dataset_file):
    """Test DatasetConfig.load dies when path does not exist."""
    dataset_file.unlink()
    with pytest.raises(FileNotFoundError, match=f'.*{dataset_file}.*'):
        dataset_config.load(path=dataset_file)
