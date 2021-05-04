"""Unit tests for lv/models/embeddings module."""
from lv.models import embeddings
from lv.utils import lang

import numpy
import pytest
import spacy

VOCAB = ('dog', 'cat', 'mouse')
VECTOR_SIZE = 99


@pytest.fixture(scope='module')
def nlp():
    """Return a single spacy instance for testing."""
    nlp = spacy.load('en_core_web_sm')
    for word in VOCAB:
        nlp.vocab.set_vector(word, numpy.random.randn(VECTOR_SIZE))
    return nlp


@pytest.fixture
def indexer():
    """Return an indexer for testing."""
    return lang.indexer(VOCAB)


def test_spacy(indexer, nlp):
    """Test spacy returns well formed embeddings."""
    actual = embeddings.spacy(indexer, nlp=nlp)
    assert actual.num_embeddings == len(indexer)
    assert actual.embedding_dim == VECTOR_SIZE
    assert not actual.weight.data[:len(VOCAB)].eq(0).all(dim=-1).any()
    assert actual.weight.data[len(VOCAB):].eq(0).all()


def test_spacy_no_vectors(indexer):
    """Test spacy dies when no vectors present."""
    with pytest.raises(ValueError, match='.*no vectors.*'):
        embeddings.spacy(indexer, nlp=spacy.load('en_core_web_sm'))
