"""Unit tests for lv/models/featurizers module."""
from lv.models import featurizers

import pytest
import torch


def test_pretrained_pyramid_featurizer_init_bad_config():
    """Test PretrainedPyramidFeaturizer.__init__ dies on bad config."""
    bad = 'bad-config'
    with pytest.raises(ValueError, match=f'.*{bad}.*'):
        featurizers.PretrainedPyramidFeaturizer(config=bad)


BATCH_SIZE = 10
IMAGE_SIZE = 224
IMAGE_SHAPE = (3, IMAGE_SIZE, IMAGE_SIZE)
MASK_SHAPE = (1, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def images():
    """Return fake images for testing."""
    return torch.rand(BATCH_SIZE, *IMAGE_SHAPE)


@pytest.fixture
def masks():
    """Return fake masks for testing."""
    return torch.randint(2, size=(BATCH_SIZE, *MASK_SHAPE))


@pytest.mark.parametrize('config', ('resnet18', 'alexnet'))
def test_pretrained_pyramid_featurizer_forward(config, images, masks):
    """Test PretrainedPyramidFeaturizer.forward returns correct shape."""
    featurizer = featurizers.PretrainedPyramidFeaturizer(config=config,
                                                         pretrained=False)
    actual = featurizer(images, masks)
    assert actual.shape == (BATCH_SIZE, featurizer.feature_size)
    assert not torch.isnan(actual).any()


@pytest.mark.parametrize('config', ('resnet18', 'alexnet'))
def test_pretrained_pyramid_featurizer_forward_invalid_mask(
        config, images, masks):
    """Test PretrainedPyramidFeaturizer.forward handles some invalid masks."""
    featurizer = featurizers.PretrainedPyramidFeaturizer(config=config,
                                                         pretrained=False)
    masks[-2:] = 0
    actual = featurizer(images, masks)
    assert actual.shape == (BATCH_SIZE, featurizer.feature_size)
    assert actual[-2:].eq(0).all()
    assert not actual[:-2].eq(0).all()
    assert not torch.isnan(actual).any()


@pytest.mark.parametrize('config', ('resnet18', 'alexnet'))
def test_pretrained_pyramid_featurizer_forward_all_invalid_masks(
        config, images, masks):
    """Test PretrainedPyramidFeaturizer.forward handles all invalid masks."""
    featurizer = featurizers.PretrainedPyramidFeaturizer(config=config,
                                                         pretrained=False)
    actual = featurizer(images, torch.zeros_like(masks))
    assert actual.shape == (BATCH_SIZE, featurizer.feature_size)
    assert actual.eq(0).all()
