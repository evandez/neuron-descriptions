"""Unit tests for `src.utils.lang` module."""
import collections
import itertools

from src.utils import lang

import pytest
import spacy


@pytest.fixture(scope='module')
def nlp():
    """Return a single spacy instance for testing."""
    return spacy.load('en_core_web_sm')


def test_tokenizer():
    """Test tokenizer factory sets state correctly."""
    tokenizer = lang.tokenizer()
    assert tokenizer.nlp is not None
    assert tokenizer.ignore_stop is True
    assert tokenizer.ignore_punct is True
    assert tokenizer.lemmatize is True


@pytest.mark.parametrize('lemmatize,lowercase,ignore_punct,ignore_stop',
                         itertools.product((False, True), repeat=4))
def test_tokenizer_override(nlp, lemmatize, lowercase, ignore_punct,
                            ignore_stop):
    """Test tokenizer factory supports overriding flags."""
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


TOKEN_0 = 'foo'
TOKEN_1 = 'bar'
TOKEN_2 = 'baz'
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
    (0, True),
    (1, True),
    (2, True),
    ('foob', False),
    (3, False),
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


@pytest.fixture
def indexer(vocab, tokenizer):
    """Return an indexer for testing."""
    return lang.Indexer(vocab, tokenizer)


START_INDEX = 3
STOP_INDEX = 4
PAD_INDEX = 5
UNK_INDEX = 6


def test_indexer_start_index(indexer):
    """Test Indexer.start_index returns first index after vocab length."""
    assert indexer.start_index == START_INDEX


def test_indexer_stop_index(indexer):
    """Test Indexer.stop_index returns second index after vocab length."""
    assert indexer.stop_index == STOP_INDEX


def test_indexer_pad_index(indexer):
    """Test Indexer.pad_index returns third index after vocab length."""
    assert indexer.pad_index == PAD_INDEX


def test_indexer_unk_index(indexer):
    """Test Indexer.unk_index returns fourth index after vocab length."""
    assert indexer.unk_index == UNK_INDEX


def test_indexer_specials(indexer):
    """Test Indexer.specials returns all special tokens."""
    assert indexer.specials == collections.OrderedDict((
        (START_INDEX, lang.START_TOKEN),
        (STOP_INDEX, lang.STOP_TOKEN),
        (PAD_INDEX, lang.PAD_TOKEN),
        (UNK_INDEX, lang.UNK_TOKEN),
    ))


def test_indexer_tokens(indexer):
    """Test indexer.tokens returns all tokens in order."""
    assert indexer.tokens == (*TOKENS, *(lang.START_TOKEN, lang.STOP_TOKEN,
                                         lang.PAD_TOKEN, lang.UNK_TOKEN))


def test_indexer_ids(indexer):
    """Test indexer.ids returns correct ID mapping."""
    assert indexer.ids == {
        TOKEN_0: 0,
        TOKEN_1: 1,
        TOKEN_2: 2,
        lang.START_TOKEN: START_INDEX,
        lang.STOP_TOKEN: STOP_INDEX,
        lang.PAD_TOKEN: PAD_INDEX,
        lang.UNK_TOKEN: UNK_INDEX,
    }


def test_indexer_unique(indexer):
    """Test Indexer.unique returns set of unique tokens."""
    assert indexer.unique == set(TOKENS) | {
        lang.START_TOKEN,
        lang.STOP_TOKEN,
        lang.PAD_TOKEN,
        lang.UNK_TOKEN,
    }


@pytest.mark.parametrize('key,expected', (
    (TOKEN_1, 1),
    (1, TOKEN_1),
    (slice(0, 2), (TOKEN_0, TOKEN_1)),
    (lang.START_TOKEN, START_INDEX),
    (PAD_INDEX, lang.PAD_TOKEN),
))
def test_indexer_getitem(indexer, key, expected):
    """Test Indexer.__getitem__ handles str/int/slice inputs."""
    actual = indexer[key]
    assert actual == expected


def test_indexer_len(indexer):
    """Test Indexer.__len__ returns number of indexable tokens."""
    assert len(indexer) == len(TOKENS) + 4


@pytest.mark.parametrize('token,expected', (
    (TOKEN_1, True),
    (1, True),
    (lang.START_TOKEN, True),
    (START_INDEX, True),
    ('foob', False),
    (10, False),
))
def test_indexer_contains(indexer, token, expected):
    """Test Indexer.__contains__ returns True if it knows about token."""
    actual = token in indexer
    assert actual is expected


@pytest.mark.parametrize(
    'init_kwargs,call_kwargs,texts,expected',
    (
        # Basic use cases.
        ({}, {}, 'baz bar foo', (2, 1, 0)),
        ({}, {}, ('baz bar foo',), ((2, 1, 0),)),
        ({}, {}, ('baz bar foo', 'baz'), ((2, 1, 0), (2,))),
        ({}, {}, ('baz bar foo', 'baz bad'), ((2, 1, 0), (2,))),

        # Configuration at init time.
        (dict(start=True), {}, 'baz bar foo', (3, 2, 1, 0)),
        (dict(stop=True), {}, 'baz bar foo', (2, 1, 0, 4)),
        (dict(pad=True), {}, 'baz bar foo', (2, 1, 0)),
        (dict(pad=True), {}, ('baz bar foo', 'baz'), ((2, 1, 0), (2, 5, 5))),
        (dict(unk=True), {}, 'baz bar foo', (2, 1, 0)),
        (dict(unk=True), {}, 'baz bad foo', (2, 6, 0)),
        (dict(length=2), {}, ('baz bar foo', 'baz'), ((2, 1), (2,))),
        (
            dict(length=2, pad=True),
            {},
            ('baz bar foo', 'baz'),
            ((2, 1), (2, 5)),
        ),
        (
            dict(start=True, stop=True, pad=True, unk=True, length=2),
            {},
            ('baz bad foo', 'baz'),
            ((3, 2, 6, 4), (3, 2, 4, 5)),
        ),

        # Configuration at call time.
        ({}, dict(start=True), 'baz bar foo', (3, 2, 1, 0)),
        ({}, dict(stop=True), 'baz bar foo', (2, 1, 0, 4)),
        ({}, dict(pad=True), 'baz bar foo', (2, 1, 0)),
        ({}, dict(pad=True), ('baz bar foo', 'baz'), ((2, 1, 0), (2, 5, 5))),
        ({}, dict(unk=True), 'baz bar foo', (2, 1, 0)),
        ({}, dict(unk=True), 'baz bad foo', (2, 6, 0)),
        ({}, dict(length=2), ('baz bar foo', 'baz'), ((2, 1), (2,))),
        (
            {},
            dict(length=2, pad=True),
            ('baz bar foo', 'baz'),
            ((2, 1), (2, 5)),
        ),
        (
            {},
            dict(start=True, stop=True, pad=True, unk=True, length=2),
            ('baz bad foo', 'baz'),
            ((3, 2, 6, 4), (3, 2, 4, 5)),
        ),

        # Override init configuration.
        (
            dict(start=True),
            dict(start=False),
            'baz bar foo',
            (2, 1, 0),
        ),
        (
            dict(stop=True),
            dict(stop=False),
            'baz bar foo',
            (2, 1, 0),
        ),
        (
            dict(pad=True),
            dict(pad=False),
            ('baz bar foo', 'baz'),
            ((2, 1, 0), (2,)),
        ),
        (
            dict(unk=True),
            dict(unk=False),
            'baz bad foo',
            (2, 0),
        ),
        (
            dict(length=2),
            dict(length=3),
            ('baz bar foo', 'baz'),
            ((2, 1, 0), (2,)),
        ),
    ))
def test_indexer_call(vocab, tokenizer, init_kwargs, call_kwargs, texts,
                      expected):
    """Test Indexer.__call__ correctly indexes inputs.

    This function also tests `Indexer.index` implicitly.
    """
    indexer = lang.Indexer(vocab, tokenizer, **init_kwargs)
    actual = indexer(texts, **call_kwargs)
    assert actual == expected


def test_indexer_index_empty(indexer):
    """Test Indexer.index handles empty inputs."""
    assert indexer.index(()) == ()


@pytest.mark.parametrize(
    'kwargs,indexed,expected',
    (
        # Basic cases.
        ({}, (0, 1), ('foo', 'bar')),
        ({}, ((0, 1), (1, 2)), (('foo', 'bar'), ('bar', 'baz'))),
        (
            {},
            ((3, 0, 1, 2, 4), (0, 0, 6, 4, 5)),
            (
                ('<start>', 'foo', 'bar', 'baz', '<stop>'),
                ('foo', 'foo', '<unk>', '<stop>', '<pad>'),
            ),
        ),

        # Overrides.
        (
            dict(specials=False),
            ((3, 0, 1, 2, 4), (0, 0, 6, 4, 5)),
            (
                ('foo', 'bar', 'baz'),
                ('foo', 'foo'),
            ),
        ),
        (
            dict(start=False),
            ((3, 0, 1, 2, 4), (0, 0, 6, 4, 5)),
            (
                ('foo', 'bar', 'baz', '<stop>'),
                ('foo', 'foo', '<unk>', '<stop>', '<pad>'),
            ),
        ),
        (
            dict(stop=False),
            ((3, 0, 1, 2, 4), (0, 0, 6, 4, 5)),
            (
                ('<start>', 'foo', 'bar', 'baz'),
                ('foo', 'foo', '<unk>', '<pad>'),
            ),
        ),
        (
            dict(pad=False),
            ((3, 0, 1, 2, 4), (0, 0, 6, 4, 5)),
            (
                ('<start>', 'foo', 'bar', 'baz', '<stop>'),
                ('foo', 'foo', '<unk>', '<stop>'),
            ),
        ),
        (
            dict(unk=False),
            ((3, 0, 1, 2, 4), (0, 0, 6, 4, 5)),
            (
                ('<start>', 'foo', 'bar', 'baz', '<stop>'),
                ('foo', 'foo', '<stop>', '<pad>'),
            ),
        ),
    ))
def test_indexer_unindex(indexer, kwargs, indexed, expected):
    """Test Indexer.unindex correctly unindexes."""
    actual = indexer.unindex(indexed, **kwargs)
    assert actual == expected


def test_indexer_unindex_empty(indexer):
    """Test Indexer.unindex handles empty inputs."""
    assert indexer.unindex(()) == ()


def test_indexer_unindex_bad_index(indexer):
    """Test Indexer.undo dies when given bad index."""
    bad = 100
    with pytest.raises(ValueError, match=f'.*{bad}.*'):
        indexer.unindex((0, bad))


def test_indexer():
    """Test indexer factory sets defaults correctly."""
    indexer = lang.indexer(TOKENS)
    assert len(indexer) == len(TOKENS) + 4


def test_indexer_defaults(tokenizer):
    """Test indexer factory sets defaults correctly."""
    indexer = lang.indexer(TOKENS, tokenize=tokenizer)
    assert len(indexer) == len(TOKENS) + 4
    assert indexer.tokenize is tokenizer


@pytest.mark.parametrize('texts,expected', (
    (['foo', 'bar'], 'foo bar'),
    (('annotation 1', 'annotation 2'), ('annotation 1 annotation 2')),
    ({'a', 'text'}, 'a text'),
    (frozenset({'foo', 'bar'}), 'bar foo'),
    ('this is a test', 'this is a test'),
))
def test_join(texts, expected):
    """Test join correctly joins texts."""
    actual = lang.join(texts)
    assert actual == expected


def test_join_bad_input():
    """Test join dies when not given string or iterable of strings."""
    with pytest.raises(ValueError, match='.*dict.*'):
        lang.join({'foo': 'bar'})
