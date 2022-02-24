"""Unit tests `src.exemplars.models`."""
import collections

from src.exemplars import models

import pytest
from torch import nn


@pytest.mark.parametrize('kwargs', (
    dict(),
    dict(k=5),
    dict(k=5, batch_size=64),
    dict(k=5, quantile=.95, output_size=256, batch_size=64, image_size=224),
))
def test_model_exemplars_config_kwargs(kwargs):
    """Test ModelExemplarsConfig.kwargs returns all kwargs."""
    config = models.ModelExemplarsConfig(**kwargs)
    assert config.kwargs == kwargs


def test_generative_model_exemplars_config_post_init():
    """Test GenerativeModelExemplarsConfig.__post_init__ validates."""
    with pytest.raises(ValueError, match='.*requires dataset.*'):
        models.GenerativeModelExemplarsConfig()


@pytest.mark.parametrize('kwargs', (
    dict(),
    dict(k=5),
    dict(k=5, batch_size=64),
    dict(k=5, quantile=.95, output_size=256, batch_size=64, image_size=224),
))
def test_generative_model_exemplars_config_kwargs(kwargs):
    """Test GenerativeModelExemplarsConfig.kwargs does not return dataset."""
    config = models.GenerativeModelExemplarsConfig(dataset='foo', **kwargs)
    assert config.kwargs == kwargs


class FakeModelConfig(models.ModelConfig):
    """A fake model config object."""

    def __init__(self, model, layers=None):
        """Initialize the config."""
        self.model = model
        self.layers = layers

    def load(self, *_args, **_kwargs):
        """Do nothing."""
        return self.model


MODEL_KEY = 'model'

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
def test_load(model, layers, expected):
    """Test load returns layers."""
    config = FakeModelConfig(model, layers)
    actuals = models.load(MODEL_KEY, configs={MODEL_KEY: config})
    assert len(actuals) == 3
    actual_model, actual_layers, actual_config = actuals
    assert actual_model is model
    assert tuple(actual_layers) == expected
    assert actual_config is config
