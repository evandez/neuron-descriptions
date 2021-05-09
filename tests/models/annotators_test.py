"""Unit tests for lv/models/annotators module."""
from lv.models import annotators
from lv.utils import lang
from tests import conftest as root
from tests.models import conftest as local

import pytest
import torch
from torch.utils import data

BATCH_SIZE = 2
N_TOP_IMAGES = 5
VOCAB = ('foo', 'bar', 'baz')
VOCAB_SIZE = len(VOCAB)


@pytest.fixture
def word_classifier_head():
    """Return a WordClassifierHead for testing."""
    return annotators.WordClassifierHead(local.FEATURE_SIZE, VOCAB_SIZE)


@pytest.fixture
def features():
    """Return fake features for testing."""
    return torch.randn(BATCH_SIZE, root.N_TOP_IMAGES_PER_UNIT,
                       local.FEATURE_SIZE)


def test_word_classifier_head_forward(word_classifier_head, features):
    """Test WordClassifierHead outputs correct shape/value range."""
    actual = word_classifier_head(features)
    assert actual.shape == (BATCH_SIZE, VOCAB_SIZE)
    assert actual.min() >= 0
    assert actual.max() <= 1


def test_word_annotations_post_init():
    """Test WordAnnotations.__post_init__ validates batch sizes."""
    with pytest.raises(ValueError, match=f'.*{BATCH_SIZE - 1}.*'):
        annotators.WordAnnotations(torch.rand(BATCH_SIZE, VOCAB_SIZE),
                                   [['foo']] * (BATCH_SIZE - 1),
                                   [[0]] * (BATCH_SIZE - 1))


@pytest.fixture(scope='module')
def indexer():
    """Return a fake Indexer for testing."""
    return lang.indexer(VOCAB)


@pytest.fixture
def word_annotator(indexer, featurizer, word_classifier_head):
    """Return a WordAnnotator for testing."""
    return annotators.WordAnnotator(indexer, featurizer, word_classifier_head)


def test_word_annotator_feature_size(word_annotator):
    """Test WordAnnotator.feature_size returns correct size."""
    assert word_annotator.feature_size == local.FEATURE_SIZE


def test_word_annotator_vocab_size(word_annotator):
    """Test WordAnnotator.vocab_size returns correct size."""
    assert word_annotator.vocab_size == len(VOCAB)


@pytest.fixture
def images(top_image_tensors):
    """Return fake images for testing."""
    return top_image_tensors[0]


@pytest.fixture
def masks(top_image_masks):
    """Return fake masks for testing."""
    return top_image_masks[0]


def assert_word_annotations_valid(actual, batch_size):
    """Assert the given word annotations are valid."""
    assert actual.probabilities.shape == (batch_size, VOCAB_SIZE)

    assert len(actual.words) == batch_size
    for words in actual.words:
        assert all(isinstance(word, str) for word in words)

    assert len(actual.indices) == batch_size
    for indices in actual.indices:
        assert all(isinstance(index, int) for index in indices)


def test_word_annotator_forward_images_and_masks(word_annotator, images,
                                                 masks):
    """Test WordAnnotator.forward handles image and mask inputs."""
    actual = word_annotator(images, masks)
    assert_word_annotations_valid(actual, len(images))


@pytest.fixture
def dataset(annotated_top_images_dataset):
    """Return a small AnnotatedTopImagesDataset for testing."""
    return data.Subset(annotated_top_images_dataset, tuple(range(5)))


def test_word_annotator_predict(word_annotator, dataset):
    """Test WordAnnotator.predict can process entire dataset."""
    actual = word_annotator.predict(dataset,
                                    batch_size=BATCH_SIZE,
                                    display_progress=False)
    assert_word_annotations_valid(actual, len(dataset))


def test_word_annotator_score(word_annotator, dataset):
    """Test WordAnnotator.score can process entire dataset."""
    actual = word_annotator.score(dataset,
                                  batch_size=BATCH_SIZE,
                                  display_progress=False)
    assert len(actual) == 2
    f1, predictions = actual
    assert f1 >= 0 and f1 <= 1
    assert_word_annotations_valid(predictions, len(dataset))


def test_word_annotator_fit(dataset, featurizer):
    """Test WordAnnotator.fit can train from start to finish."""
    actual = annotators.WordAnnotator.fit(
        dataset,
        featurizer,
        max_epochs=1,
        batch_size=BATCH_SIZE,
        optimizer_kwargs={'lr': 1e-4},
        indexer_kwargs={'ignore_in': ('foo',)},
        display_progress=False)
    # Just some basic assertions...
    assert actual.indexer is not None
    assert actual.featurizer is featurizer
    assert actual.classifier.feature_size == local.FEATURE_SIZE
    assert not torch.isnan(actual.classifier.classifier.weight.data).any()
