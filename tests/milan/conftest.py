"""Test configuration shared by all `src.models` tests."""
from src.milan import encoders

import numpy
import pytest
import torch


class FakeEncoder(encoders.Encoder):
    """A fake Encoder that always returns zeros."""

    def __init__(self, feature_shape):
        """Initialize the encoder."""
        super().__init__()
        self.feature_shape = feature_shape

    def forward(self, images, masks, **kwargs):
        """Assert on inputs and return zeros."""
        assert not kwargs
        assert images.shape[0] == masks.shape[0]
        assert images.shape[2:] == masks.shape[2:]
        assert images.shape[1] == 3
        assert masks.shape[1] == 1
        return torch.zeros(len(images), *self.feature_shape)


FEATURE_SHAPE = (10, 10)
FEATURE_SIZE = numpy.prod(FEATURE_SHAPE)


@pytest.fixture
def encoder():
    """Return a FakeEncoder for testing."""
    return FakeEncoder(FEATURE_SHAPE)
