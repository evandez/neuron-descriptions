"""Unit tests for `src.utils.hubs` module."""
import collections
import pathlib
import tempfile

from src.utils import hubs

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
    return hubs.ModelConfig(factory=Model, flag=True)


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
    model_config = hubs.ModelConfig(factory=Model,
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
    model_config = hubs.ModelConfig(Model,
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
    return hubs.DatasetConfig(factory=Dataset, flag=True)


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
    dataset_config = hubs.DatasetConfig(factory=Dataset,
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


# loaders_test below this


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
        return self.model


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


MODEL_KEY = 'my-model'


def test_model_hub_load(model, model_path):
    """Test model correctly reads config."""
    hub = hubs.ModelHub(
        **{
            MODEL_KEY: FakeModelConfig(model, model_path, flag=True),
            'bad': DangerConfig()
        })
    actual = hub.load(MODEL_KEY, path=model_path, flag=True)
    assert actual is model


def test_model_hub_load_no_path(model):
    """Test model correctly reads config when no path given."""
    hub = hubs.ModelHub(
        **{
            MODEL_KEY:
                FakeModelConfig(model,
                                pathlib.Path(__file__).parents[2] / 'models' /
                                f'{MODEL_KEY}.pth',
                                flag=True),
            'bad':
                DangerConfig()
        })
    actual = hub.load(MODEL_KEY, flag=True)
    assert actual is model


def test_model_hub_load_bad_keys(model):
    """Test model dies on bad keys."""
    hub = hubs.ModelHub(**{
        MODEL_KEY: FakeModelConfig(model),
    })
    with pytest.raises(KeyError, match='.*bad.*'):
        hub.load('bad')


class FakeDatasetConfig:
    """A fake dataset config object."""

    def __init__(self, dataset, *args, requires_path=True, **kwargs):
        """Initialize the dataset config."""
        self.dataset = dataset
        self.requires_path = requires_path
        self.args = args
        self.kwargs = kwargs

    def load(self, *args, **kwargs):
        """Fake load that just assert args and kwargs are correct."""
        assert args == self.args
        assert kwargs == self.kwargs
        return self.dataset


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


DATASET_KEY = 'my-dataset'


def test_dataset_hub_load(dataset, dataset_path):
    """Test dataset correctly reads config."""
    hub = hubs.DatasetHub(
        **{
            DATASET_KEY:
                FakeDatasetConfig(dataset, path=dataset_path, flag=True),
            'bad':
                DangerConfig()
        })
    actual = hub.load(DATASET_KEY, path=dataset_path, flag=True)
    assert actual is dataset


def test_dataset_hub_load_no_path(dataset):
    """Test DatasetHub.load correctly reads config when no path provided."""
    hub = hubs.DatasetHub(
        **{
            DATASET_KEY:
                FakeDatasetConfig(dataset,
                                  path=pathlib.Path(__file__).parents[2] /
                                  'data' / DATASET_KEY,
                                  flag=True),
            'bad':
                DangerConfig()
        })
    actual = hub.load(DATASET_KEY, flag=True)
    assert actual is dataset


def test_dataset_hub_load_no_path_no_requires_path(dataset):
    """Test DatasetHub.load correctly reads config when requires_path=False."""
    expected = FakeDatasetConfig(dataset,
                                 requires_path=False,
                                 path=None,
                                 flag=True)
    hub = hubs.DatasetHub(**{DATASET_KEY: expected, 'bad': DangerConfig()})
    actual = hub.load(DATASET_KEY, flag=True)
    assert actual is dataset


def test_dataset_bad_key(dataset):
    """Test dataset dies on bad keys."""
    hub = hubs.DatasetHub(**{
        DATASET_KEY: FakeDatasetConfig(dataset, path=None, flag=True),
    })
    with pytest.raises(KeyError, match='.*bad.*'):
        hub.load('bad')
