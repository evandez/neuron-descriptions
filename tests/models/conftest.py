"""Test configuration shared by all lv/models tests."""
from lv.models import featurizers

import numpy
import pytest
import torch


class FakeFeaturizer(featurizers.Featurizer):
    """A fake Featurizer that always returns zeros."""

    def __init__(self, feature_shape):
        """Initialize the featurizer."""
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
def featurizer():
    """Return a FakeFeaturizer for testing."""
    return FakeFeaturizer(FEATURE_SHAPE)
