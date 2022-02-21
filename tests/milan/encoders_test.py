"""Unit tests for `src.models.encoders` module."""
from src.milan import encoders
from tests import conftest as root
from tests.milan import conftest as local

import pytest
import torch


@pytest.mark.parametrize('device', (None, 'cpu', torch.device('cpu')))
def test_encoder_map(encoder, top_images_dataset, device):
    """Test Encoder.map returns TensorDataset of right size."""
    actual = encoder.map(top_images_dataset,
                         image_index=-2,
                         mask_index=-1,
                         display_progress_as=None,
                         device=device)
    assert len(actual) == len(top_images_dataset)
    for (features,) in actual:
        assert features.shape == (root.N_TOP_IMAGES_PER_UNIT,
                                  *local.FEATURE_SHAPE)
        assert features.eq(0).all()


def test_pyramid_conv_encoder_init_bad_config():
    """Test PyramidConvEncoder.__init__ dies on bad config."""
    bad = 'bad-config'
    with pytest.raises(ValueError, match=f'.*{bad}.*'):
        encoders.PyramidConvEncoder(config=bad)


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
    return torch.randint(2, size=(BATCH_SIZE, *MASK_SHAPE), dtype=torch.float)


@pytest.mark.parametrize('config', ('resnet18', 'alexnet'))
def test_pyramid_conv_encoder_forward(config, images, masks):
    """Test PyramidConvEncoder.forward returns correct shape."""
    encoder = encoders.PyramidConvEncoder(config=config, pretrained=False)
    actual = encoder(images, masks)
    assert actual.shape == (BATCH_SIZE, *encoder.feature_shape)
    assert not torch.isnan(actual).any()


@pytest.mark.parametrize('config', ('resnet18', 'alexnet'))
def test_pyramid_conv_encoder_forward_invalid_mask(config, images, masks):
    """Test PyramidConvEncoder.forward handles some invalid masks."""
    encoder = encoders.PyramidConvEncoder(config=config, pretrained=False)
    masks[-2:] = 0
    actual = encoder(images, masks)
    assert actual.shape == (BATCH_SIZE, *encoder.feature_shape)
    assert actual[-2:].eq(0).all()
    assert not actual[:-2].eq(0).all()
    assert not torch.isnan(actual).any()


@pytest.mark.parametrize('config', ('resnet18', 'alexnet'))
def test_pyramid_conv_encoder_forward_all_invalid_masks(config, images, masks):
    """Test PyramidConvEncoder.forward handles all invalid masks."""
    encoder = encoders.PyramidConvEncoder(config=config, pretrained=False)
    actual = encoder(images, torch.zeros_like(masks))
    assert actual.shape == (BATCH_SIZE, *encoder.feature_shape)
    assert actual.eq(0).all()
