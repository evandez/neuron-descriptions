"""Unit tests for lv/utils/lang module."""
import itertools

from lv.utils import lang

import pytest
import spacy


@pytest.fixture(scope='module')
def nlp():
    """Return a single spacy instance for testing."""
    return spacy.load('en_core_web_sm')


def test_tokenizer():
    """Test tokenizer sets state correctly."""
    tokenizer = lang.tokenizer()
    assert tokenizer.nlp is not None
    assert tokenizer.ignore_stop is True
    assert tokenizer.ignore_punct is True
    assert tokenizer.lemmatize is True


@pytest.mark.parametrize('lemmatize,lowercase,ignore_punct,ignore_stop',
                         itertools.product((False, True), repeat=4))
def test_tokenizer_override(nlp, lemmatize, lowercase, ignore_punct,
                            ignore_stop):
    """Test tokenizer supports overriding flags."""
    tokenizer = lang.tokenizer(nlp=nlp,
                               lemmatize=lemmatize,
                               lowercase=lowercase,
                               ignore_punct=ignore_punct,
                               ignore_stop=ignore_stop)
    assert tokenizer.nlp is nlp
    assert tokenizer.lemmatize is lemmatize
    assert tokenizer.lowercase is lowercase
    assert tokenizer.ignore_stop is ignore_stop
    assert tokenizer.ignore_punct is ignore_punct


@pytest.mark.parametrize('kwargs,texts,expected', (
    ({}, 'the Foo bar broke.', ('foo', 'bar', 'break')),
    ({}, ('the Foo bar broke.',), (('foo', 'bar', 'break'),)),
    (
        dict(lemmatize=False),
        ('the Foo bar stayed.',),
        (('foo', 'bar', 'stayed'),),
    ),
    (dict(lowercase=False), 'the Foo bar.', ('Foo', 'bar')),
    (dict(ignore_punct=False), 'the Foo bar.', ('foo', 'bar', '.')),
    (dict(ignore_stop=False), 'the Foo bar.', ('the', 'foo', 'bar')),
))
def test_tokenizer_call(nlp, kwargs, texts, expected):
    """Test Tokenizer.__call__ correctly tokenizes sentences."""
    tokenizer = lang.tokenizer(nlp=nlp, **kwargs)
    actual = tokenizer(texts)
    assert actual == expected


TOKEN_0 = 'a'
TOKEN_1 = 'b'
TOKEN_2 = 'c'
TOKENS = (TOKEN_0, TOKEN_1, TOKEN_2)


@pytest.fixture
def vocab():
    """Return a Vocab for testing."""
    return lang.Vocab(TOKENS)


@pytest.mark.parametrize(
    'key,expected',
    (
        (1, TOKEN_1),
        (TOKEN_1, 1),
        (slice(0, 2), (TOKEN_0, TOKEN_1)),
    ),
)
def test_vocab_getitem(vocab, key, expected):
    """Test Vocab.__getitem__ handles ints, strings, and slices."""
    actual = vocab[key]
    assert actual == expected


def test_vocab_len(vocab):
    """Test Vocab.__len__ returns correct length."""
    assert len(vocab) == len(TOKENS)


@pytest.mark.parametrize('token,expected', (
    (TOKEN_0, True),
    (TOKEN_1, True),
    (TOKEN_2, True),
    ('foo', False),
))
def test_vocab_contains(vocab, token, expected):
    """Test Vocab.__contains__ correctly checks if token is in vocab."""
    actual = token in vocab
    assert actual is expected


def test_vocab_ids(vocab):
    """Test Vocab.ids correctly maps integer indices to strings."""
    assert vocab.ids == {
        TOKEN_0: 0,
        TOKEN_1: 1,
        TOKEN_2: 2,
    }


def test_vocab_unique(vocab):
    """Test Vocab.unique returns all unique tokens."""
    assert vocab.unique == frozenset(TOKENS)


@pytest.fixture
def tokenizer(nlp):
    """Return a Tokenizer for testing."""
    return lang.tokenizer(nlp=nlp)


TEXT_A = 'The Foo bar ran.'
TEXT_B = 'A Foo Bar ran wildly.'
TEXTS = (TEXT_A, TEXT_B)


@pytest.mark.parametrize('kwargs,expected', (
    ({}, ('foo', 'bar', 'run', 'wildly')),
    (dict(ignore_in=('foo',)), ('bar', 'run', 'wildly')),
    (dict(ignore_rarer_than=1), ('foo', 'bar', 'run')),
))
def test_vocab(tokenizer, kwargs, expected):
    """Test vocab factory in basic cases."""
    vocab = lang.vocab(TEXTS, tokenize=tokenizer, **kwargs)
    assert vocab.tokens == expected


def test_vocab_default_tokenizer():
    """Test vocab factory creates default tokenizer when necessary."""
    vocab = lang.vocab(TEXTS)
    assert vocab.tokens
