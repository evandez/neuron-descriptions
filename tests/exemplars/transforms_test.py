"""Unit tests for `src.exemplars.transforms` module."""
from src.exemplars import transforms

import pytest
import torch


@pytest.fixture
def tensors():
    """Return some tensor data for testing."""
    return torch.randn(10, 10)


@pytest.mark.parametrize('device', (None, 'cpu', torch.device('cpu')))
def test_map_location(tensors, device):
    """Test map_location handles all kinds of devices."""
    args = ('foo', tensors)
    actuals = transforms.map_location(args, device)
    assert len(actuals) == 2
    actual_str, actual_tensors = actuals
    assert actual_str == 'foo'
    assert actual_tensors.allclose(tensors)


def test_first():
    """Test first returns first arg as tuple."""
    actual = transforms.first('foo', 'bar')
    assert actual == ('foo',)


def test_identity():
    """Test identity returns single argument."""
    actual = transforms.identity('foo')
    assert actual == 'foo'


def test_identities():
    """Test identities returns all arguments."""
    args = ('foo', 'bar')
    actual = transforms.identities(*args)
    assert actual == args


VIT_BATCH_SIZE = 32
VIT_SPATIAL_SIZE = 28
VIT_N_PATCHES = VIT_SPATIAL_SIZE**2 + 1
VIT_N_UNITS = 3072


def test_spatialize_vit_mlp():
    """Test spatialize_vit_mlp correctly reshapes vit features."""
    hiddens = torch.randn(VIT_BATCH_SIZE, VIT_N_PATCHES, VIT_N_UNITS)
    actual = transforms.spatialize_vit_mlp(hiddens)
    assert actual.shape == (VIT_BATCH_SIZE, VIT_N_UNITS, VIT_SPATIAL_SIZE,
                            VIT_SPATIAL_SIZE)
