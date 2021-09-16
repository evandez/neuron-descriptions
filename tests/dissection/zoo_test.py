"""Unit tests for dissection zoo."""
import collections

from lv.dissection import zoo

import pytest
from torch import nn


@pytest.mark.parametrize('kwargs', (
    dict(),
    dict(k=5),
    dict(k=5, batch_size=64),
    dict(k=5, quantile=.95, output_size=256, batch_size=64, image_size=224),
))
def test_model_dissection_config_kwargs(kwargs):
    """Test ModelDissectionConfig.kwargs returns all kwargs."""
    config = zoo.ModelDissectionConfig(**kwargs)
    assert config.kwargs == kwargs


def test_generative_model_dissection_config_post_init():
    """Test GenerativeModelDissectionConfig.__post_init__ validates."""
    with pytest.raises(ValueError, match='.*requires dataset.*'):
        zoo.GenerativeModelDissectionConfig()


@pytest.mark.parametrize('kwargs', (
    dict(),
    dict(k=5),
    dict(k=5, batch_size=64),
    dict(k=5, quantile=.95, output_size=256, batch_size=64, image_size=224),
))
def test_generative_model_dissection_config_kwargs(kwargs):
    """Test GenerativeModelDissectionConfig.kwargs does not return dataset."""
    config = zoo.GenerativeModelDissectionConfig(dataset='foo', **kwargs)
    assert config.kwargs == kwargs


class FakeModelConfig(zoo.ModelConfig):
    """A fake model config object."""

    def __init__(self, model, layers=None):
        """Initialize the config."""
        self.model = model
        self.layers = layers

    def load(self, *_args, **_kwargs):
        """Do nothing."""
        return self.model


MODEL_NAME = 'model'
DATASET_NAME = 'dataset'

LAYER_1 = 'layer_1'
LAYER_2 = 'layer_2'


@pytest.fixture
def model():
    """Return a fake model for testing."""
    return nn.Sequential(
        collections.OrderedDict([
            (LAYER_1, nn.Linear(10, 10)),
            (LAYER_2, nn.Linear(10, 20)),
        ]))


@pytest.mark.parametrize('layers,expected', (
    (None, (LAYER_1, LAYER_2)),
    ((LAYER_1,), (LAYER_1,)),
))
def test_model(model, layers, expected):
    """Test model loader returns layers."""
    config = FakeModelConfig(model, layers)
    configs = {MODEL_NAME: {DATASET_NAME: config}}
    actuals = zoo.model(MODEL_NAME, DATASET_NAME, source=configs)
    assert len(actuals) == 3
    actual_model, actual_layers, actual_config = actuals
    assert actual_model is model
    assert tuple(actual_layers) == expected
    assert actual_config is config
