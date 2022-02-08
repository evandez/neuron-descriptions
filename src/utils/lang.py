"""Utilities for preprocessing language data."""
import collections
import dataclasses
import functools
from typing import Any, Mapping, Optional, Sequence, Union, cast, overload

from src.utils import serialize
from src.utils.typing import StrIterable, StrSequence, StrSet

import spacy
from spacy.lang import en


@dataclasses.dataclass(frozen=True)
class Tokenizer(serialize.Serializable):
    """A wrapper around the spacy English tokenizer supporting defaults."""

    nlp: en.English
    lemmatize: bool = True
    lowercase: bool = True
    ignore_stop: bool = True
    ignore_punct: bool = True

    @overload
    def __call__(self, texts: str) -> StrSequence:
        """Tokenize the given text.

        Args:
            texts (str): The text.

        Returns:
            StrSequence: Tokenized text.

        """
        ...

    @overload
    def __call__(self, texts: StrSequence) -> Sequence[StrSequence]:
        """Tokenize all texts.

        Args:
            texts (StrSequence): One or more texts.

        Returns:
            Sequence[StrSequence]: Token sequences for each text.

        """
        ...

    def __call__(
        self,
        texts: Union[str, StrSequence],
    ) -> Union[StrSequence, Sequence[StrSequence]]:
        """Implement both overloads."""
        tokenized = []
        for doc in self.nlp.pipe([texts] if isinstance(texts, str) else texts):
            tokens = []
            for token in doc:
                if self.ignore_stop and token.is_stop:
                    continue
                if self.ignore_punct and token.is_punct:
                    continue
                text = token.lemma_ if self.lemmatize else token.text
                text = text.lower() if self.lowercase else text
                if text.strip():
                    tokens.append(text)
            tokenized.append(tuple(tokens))

        if isinstance(texts, str):
            return tokenized[0]
        return tuple(tokenized)


def tokenizer(nlp: Optional[en.English] = None,
              lemmatize: bool = True,
              **kwargs: Any) -> Tokenizer:
    """Create a tokenizer based on a stripped spacy en_core_web_sm.

    Keyword arguments are forwarded to Tokenizer constructor.

    Args:
        nlp (Optional[en.English], optional): Spacy instance to use.
            Defaults to a stripped version of en_core_web_sm if not set.
        lemmatize (bool, optional): Whether to lemmatize tokens.
            Defaults to True.

    """
    if nlp is None:
        nlp = cast(en.English, spacy.load('en_core_web_sm'))
    return Tokenizer(nlp, lemmatize=lemmatize, **kwargs)


@dataclasses.dataclass(frozen=True)
class Vocab(serialize.Serializable):
    """A data class that stores tokens and a corresponding tokenizer."""

    tokens: StrSequence

    @overload
    def __getitem__(self, token: int) -> str:
        """Get token string for token ID.

        Args:
            token (int): The token ID.

        Returns:
            str: The token.

        """
        ...

    @overload
    def __getitem__(self, token: slice) -> StrSequence:
        """Get token strings for ID slice.

        Args:
            token (slice): The ID slice.

        Returns:
            StrSequence: The token strings.

        """
        ...

    @overload
    def __getitem__(self, token: str) -> int:
        """Get token ID for token string.

        Args:
            token (str): The token string.

        Returns:
            int: The token ID.

        """
        ...

    def __getitem__(
        self,
        token: Union[int, slice, str],
    ) -> Union[str, StrSequence, int]:
        """Get the ID/string for the given token."""
        if isinstance(token, (int, slice)):
            return self.tokens[token]
        assert isinstance(token, str)
        return self.ids[token]

    def __len__(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.tokens)

    def __contains__(self, token: Union[int, str]) -> bool:
        """Check whether vocabulary contains token or token ID.

        Args:
            token (Union[int, str]): The token or token ID.

        Returns:
            bool: True if the vocabulary contains the token.

        """
        if isinstance(token, int):
            return token >= 0 and token < len(self)
        return token in self.unique

    @functools.cached_property
    def ids(self) -> Mapping[str, int]:
        """Return mapping from word to integer ID (i.e., its index)."""
        return {token: index for index, token in enumerate(self.tokens)}

    @functools.cached_property
    def unique(self) -> StrSet:
        """Return the set of unique tokens."""
        return frozenset(self.ids)

    def properties(self) -> serialize.Properties:
        """Override `Serializable.properties`."""
        return {'tokens': self.tokens}


def vocab(texts: StrSequence,
          tokenize: Optional[Tokenizer] = None,
          ignore_rarer_than: Optional[int] = None,
          ignore_in: Optional[StrIterable] = None) -> Vocab:
    """Create vocabulary from given texts.

    All texts will be tokenized. Tokens are then cleaned and filtered.
    Remaining tokens are passed to vocabulary in order of frequency, from
    most common to least common.

    Args:
        texts (StrSequence): Texts to parse vocabulary from.
        tokenize (Optional[Tokenizer], optional): Tokenizer to use.
            Defaults to `Tokenizer.default()`.
        ignore_rarer_than (Optional[int], optional): Ignore tokens that
            appear this many times or fewer. Defaults to None.
        ignore_in (Optional[StrIterable], optional): Ignore tokens in
            this iterable. Defaults to None.

    Returns:
        Vocab: The instantiated vocabulary.

    """
    if tokenize is None:
        tokenize = tokenizer()
    if ignore_in is not None:
        ignore_in = frozenset(ignore_in)

    def ignore(token: str, count: int) -> bool:
        yn = ignore_rarer_than is not None and count <= ignore_rarer_than
        yn |= ignore_in is not None and token in ignore_in
        return yn

    tokens = [tok for toks in tokenize(texts) for tok in toks]
    counts = collections.Counter(tokens)
    tokens = [
        token for token, count in counts.most_common()
        if not ignore(token, count)
    ]

    return Vocab(tuple(tokens))


START_TOKEN = '<start>'
STOP_TOKEN = '<stop>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


@dataclasses.dataclass(frozen=True)
class Indexer(serialize.Serializable):
    """Maps string text to integer ID sequences."""

    vocab: Vocab
    tokenize: Tokenizer
    start: bool = False
    stop: bool = False
    pad: bool = False
    unk: bool = False
    length: Optional[int] = None

    @functools.cached_property
    def start_index(self) -> int:
        """Return the index of a (hypothetical) start token."""
        return len(self.vocab)

    @functools.cached_property
    def stop_index(self) -> int:
        """Return the index of a (hypothetical) stop token."""
        return len(self.vocab) + 1

    @functools.cached_property
    def pad_index(self) -> int:
        """Return index of a (hypothetical) padding token."""
        return len(self.vocab) + 2

    @functools.cached_property
    def unk_index(self) -> int:
        """Return index of a (hypothetical) unknown token."""
        return len(self.vocab) + 3

    @functools.cached_property
    def specials(self) -> Mapping[int, str]:
        """Return all special indices."""
        return collections.OrderedDict((
            (self.start_index, START_TOKEN),
            (self.stop_index, STOP_TOKEN),
            (self.pad_index, PAD_TOKEN),
            (self.unk_index, UNK_TOKEN),
        ))

    @functools.cached_property
    def tokens(self) -> StrSequence:
        """Return all tokens known by the indexer."""
        tokens = list(self.vocab.tokens)
        tokens += self.specials.values()
        return tuple(tokens)

    @functools.cached_property
    def ids(self) -> Mapping[str, int]:
        """Return a mapping from token string to ID."""
        ids = dict(self.vocab.ids)
        for index, token in self.specials.items():
            ids[token] = index
        return ids

    @functools.cached_property
    def unique(self) -> StrSet:
        """Return the set of unique tokens."""
        return frozenset(self.ids)

    @overload
    def __getitem__(self, token: int) -> str:
        """Get token string for token ID.

        Args:
            token (int): The token ID.

        Returns:
            str: The token.

        """
        ...

    @overload
    def __getitem__(self, token: slice) -> StrSequence:
        """Get token strings for ID slice.

        Args:
            token (slice): The ID slice.

        Returns:
            StrSequence: The token strings.

        """
        ...

    @overload
    def __getitem__(self, token: str) -> int:
        """Get token ID for token string.

        Args:
            token (str): The token string.

        Returns:
            int: The token ID.

        """
        ...

    def __getitem__(
        self,
        token: Union[int, slice, str],
    ) -> Union[str, StrSequence, int]:
        """Get the ID/string for the given token."""
        if isinstance(token, (int, slice)):
            return self.tokens[token]
        assert isinstance(token, str)
        return self.ids[token]

    def __len__(self) -> int:
        """Return the number of tokens in the vocabulary (include specials)."""
        return len(self.vocab) + len(self.specials)

    def __contains__(self, token: Union[int, str]) -> bool:
        """Check whether indexer contains token or token ID.

        Args:
            token (Union[int, str]): The token or token ID.

        Returns:
            bool: True if the vocabulary contains the token.

        """
        if isinstance(token, int):
            return token >= 0 and token < len(self)
        return token in self.unique

    @overload
    def __call__(self, texts: str, **kwargs: Any) -> Sequence[int]:
        """Tokenize and index the given text.

        Args:
            texts (str): The text.

        Returns:
            Sequence[int]: Indexed tokens from text.

        """
        ...

    @overload
    def __call__(self, texts: StrSequence,
                 **kwargs: Any) -> Sequence[Sequence[int]]:
        """Tokenize and index all given text.

        Args:
            texts (StrSequence): The text.

        Returns:
            Sequence[Sequence[int]]: Indexed tokens from all texts.

        """
        ...

    def __call__(
        self,
        texts: Union[str, StrSequence],
        **kwargs: Any,
    ) -> Union[Sequence[int], Sequence[Sequence[int]]]:
        """Implement all overloads."""
        tokenized = self.tokenize([texts] if isinstance(texts, str) else texts)
        indexed = self.index(tokenized, **kwargs)
        return indexed[0] if isinstance(texts, str) else indexed

    @overload
    def index(self,
              tokenized: StrSequence,
              start: Optional[bool] = ...,
              stop: Optional[bool] = ...,
              pad: Optional[bool] = ...,
              unk: Optional[bool] = ...,
              length: Optional[int] = ...) -> Sequence[int]:
        """Map the tokens to integer IDs.

        Args:
            tokenized (StrSequence): Tokens to index.
            start (Optional[bool], optional): Prepend start token if possible.
                Defaults to None.
            stop (Optional[bool], optional): Prepend stop token if possible.
                Defaults to None.
            pad (Optional[bool], optional): Pad shorter sequences with padding
                token if necessary and if possible. Defaults to None.
            unk (Optional[bool], optional): Replace unknown tokens with the
                UNK token. Otherwise, they are removed. Defaults to None.
            length (Optional[int], optional): Pad shorter sequences to this
                length, if possible, and truncate longer sequences to this
                length. This does NOT count the start and stop tokens. Defaults
                to `self.length`, or if that is not set, defaults to the length
                of the longest input sequence.

        Returns:
            Sequence[int]: The indexed sequence(s).

        """
        ...

    @overload
    def index(self,
              tokenized: Sequence[StrSequence],
              start: Optional[bool] = ...,
              stop: Optional[bool] = ...,
              pad: Optional[bool] = ...,
              unk: Optional[bool] = ...,
              length: Optional[int] = ...) -> Sequence[Sequence[int]]:
        """Tokenize the given texts and map them to integer IDs.

        Args:
            tokenized (Sequence[StrSequence]): Tokens to index.
            start (Optional[bool], optional): Prepend start token if possible.
                Defaults to None.
            stop (Optional[bool], optional): Prepend stop token if possible.
                Defaults to None.
            pad (Optional[bool], optional): Pad shorter sequences with padding
                token if necessary and if possible. Defaults to None.
            unk (Optional[bool], optional): Replace unknown tokens with the
                UNK token. Otherwise, they are removed. Defaults to None.
            length (Optional[int], optional): Pad shorter sequences to this
                length, if possible, and truncate longer sequences to this
                length. This does NOT count the start and stop tokens. Defaults
                to `self.length`, or if that is not set, defaults to the length
                of the longest input sequence.

        Returns:
            Sequence[Sequence[int]]: The indexed sequences.

        """
        ...

    def index(
        self,
        tokenized: Union[StrSequence, Sequence[StrSequence]],
        start: Optional[bool] = None,
        stop: Optional[bool] = None,
        pad: Optional[bool] = None,
        unk: Optional[bool] = None,
        length: Optional[int] = None,
    ) -> Union[Sequence[int], Sequence[Sequence[int]]]:
        """Implement all overloads."""
        if not tokenized:
            return ()

        singleton = isinstance(tokenized[0], str)
        start = self.start if start is None else start
        stop = self.stop if stop is None else stop
        pad = self.pad if pad is None else pad
        unk = self.unk if unk is None else unk
        length = length or self.length or max(len(toks) for toks in tokenized)
        for special in (start, stop):
            if special:
                length += 1

        indexed = []
        for tokens in [tokenized] if singleton else tokenized:
            tokens = cast(StrSequence, tokens)

            indices = []

            if start:
                indices.append(self.start_index)

            if unk:
                indices += [
                    self.vocab.ids.get(tok, self.unk_index) for tok in tokens
                ]
            else:
                indices += [
                    self.vocab[tok] for tok in tokens if tok in self.vocab
                ]

            if stop:
                if len(indices) >= length:
                    indices = indices[:length - 1]
                indices.append(self.stop_index)

            if len(indices) < length and pad:
                indices += [self.pad_index] * (length - len(indices))
            elif len(indices) > length:
                indices = indices[:length]

            indexed.append(tuple(indices))

        if singleton:
            return indexed[0]
        return tuple(indexed)

    @overload
    def unindex(self,
                indexed: Sequence[int],
                specials: bool = ...,
                start: bool = ...,
                stop: bool = ...,
                pad: bool = ...,
                unk: bool = ...) -> StrSequence:
        """Undo indexing on the sequence.

        Args:
            indices (Sequence[int]): The indexed sequence.
            specials (bool, optional): Include special tokens. If False,
                overrides all other options below. Defaults to True.
            start (bool, optional): Include start token. Defaults to True.
            stop (bool, optional): Include stop token. Defaults to True.
            pad (bool, optional): Include pad token. Defaults to True.
            unk (bool, optional): Include unk token. Defaults to True.

        Raises:
            ValueError: If sequence contains an unknown index.

        Returns:
            StrSequence: Unindexed sequence.

        """
        ...

    @overload
    def unindex(self,
                indexed: Sequence[Sequence[int]],
                specials: bool = ...,
                start: bool = ...,
                stop: bool = ...,
                pad: bool = ...,
                unk: bool = ...) -> Sequence[StrSequence]:
        """Undo indexing on the sequence.

        Args:
            indices (Sequence[Sequence[int]]): The indexed sequences.
            specials (bool, optional): Include special tokens. If False,
                overrides all other options below. Defaults to True.
            start (bool, optional): Include start token. Defaults to True.
            stop (bool, optional): Include stop token. Defaults to True.
            pad (bool, optional): Include pad token. Defaults to True.
            unk (bool, optional): Include unk token. Defaults to True.

        Raises:
            ValueError: If sequence contains an unknown index.

        Returns:
            Sequence[StrSequence]: Unindexed sequences.

        """
        ...

    def unindex(self,
                indexed: Union[Sequence[int], Sequence[Sequence[int]]],
                specials: bool = True,
                start: bool = True,
                stop: bool = True,
                pad: bool = True,
                unk: bool = True) -> Union[StrSequence, Sequence[StrSequence]]:
        """Undo indexing on the sequence."""
        if not indexed:
            return ()

        singleton = isinstance(indexed[0], int)

        unindexed = []
        for indices in [indexed] if singleton else indexed:
            indices = cast(Sequence[int], indices)

            tokens = []
            for index in indices:
                # First check if it's in the vocabulary.
                if index < len(self.vocab):
                    tokens.append(self.vocab[index])
                    continue

                # Then check if it's a special token.
                for (special, token), keep in zip(self.specials.items(),
                                                  (start, stop, pad, unk)):
                    if index == special:
                        if specials and keep:
                            tokens.append(token)
                        break
                else:
                    raise ValueError(f'unknown index: {index}')

            unindexed.append(tuple(tokens))

        if singleton:
            return unindexed[0]

        return tuple(unindexed)

    @overload
    def reconstruct(self, inputs: Sequence[int]) -> str:
        """Reconstruct text string from the indices.

        Args:
            inputs (Sequence[int]): The token indices.

        Returns:
            str: The text string.

        Raises:
            ValueError: If indices is empty.

        """
        ...

    @overload
    def reconstruct(self, inputs: Sequence[Sequence[int]]) -> StrSequence:
        """Reconstruct text strings from each index sequence.

        Args:
            inputs (Sequence[Sequence[int]]): The index sequences.

        Returns:
            StrSequence: The text strings.

        Raises:
            ValueError: If indices is empty.

        """
        ...

    @overload
    def reconstruct(self, inputs: StrSequence) -> str:
        """Reconstruct text string from the token sequence.

        Args:
            inputs (StrSequence): The token sequence.

        Returns:
            str: The text string.

        Raises:
            ValueError: If tokens is empty.

        """
        ...

    @overload
    def reconstruct(self, inputs: Sequence[StrSequence]) -> StrSequence:
        """Reconstruct text strings from the token sequences.

        Args:
            inputs (Sequence[StrSequence]): The token sequences.

        Returns:
            StrSequence: The text strings.

        Raises:
            ValueError: If tokens is empty.

        """
        ...

    def reconstruct(
        self,
        inputs: Union[Sequence[int], Sequence[Sequence[int]], StrSequence,
                      Sequence[StrSequence]],
    ) -> Union[str, StrSequence]:
        """Implement all overloads of `reconstruct`."""
        if not inputs:
            raise ValueError('must provide at least one seq')
        for index, item in enumerate(inputs):
            if not isinstance(item, (int, str)) and not item:
                raise ValueError(f'input seq {index} is empty')

        if isinstance(inputs[0], str):
            tokenized = cast(Sequence[StrSequence], [inputs])
        elif isinstance(inputs[0], int):
            inputs = cast(Sequence[int], inputs)
            tokenized = [self.unindex(inputs)]
        elif isinstance(inputs[0][0], str):
            tokenized = cast(Sequence[StrSequence], inputs)
        else:
            assert isinstance(inputs[0][0], int), 'unknown input type'
            inputs = cast(Sequence[Sequence[int]], inputs)
            tokenized = self.unindex(inputs)

        texts = []
        for tokens in tokenized:
            if STOP_TOKEN in tokens:
                last = tokens.index(STOP_TOKEN)
                tokens = tokens[:last]

            # First put the caption together, removing special tokens.
            text = ' '.join([
                token for token in tokens
                if token not in self.specials.values()
            ])

            # Remove spaces after punctuation.
            for token in ('.', ',', ';', ':'):
                text = text.replace(' ' + token, token)

            # ...except for dashes, which need space removed before and after.
            for token in ('-',):
                text = text.replace(' %s' % token, token)
                text = text.replace('%s ' % token, token)

            # Finally, capitalize each sentence.
            text = '. '.join([
                sentence.strip().capitalize() for sentence in text.split('.')
            ]).strip()

            texts.append(text)

        return texts[0] if isinstance(inputs[0], (str, int)) else tuple(texts)

    def properties(self, **_: Any) -> Mapping[str, Any]:
        """Override `Serializable.properties`."""
        return {
            'vocab': self.vocab,
            'tokenize': self.tokenize,
            'start': self.start,
            'stop': self.stop,
            'pad': self.pad,
            'unk': self.unk,
            'length': self.length,
        }

    @classmethod
    def resolve(cls, children: serialize.Children) -> serialize.Resolved:
        """Override `Serializable.resolve`."""
        return {'vocab': Vocab, 'tokenize': Tokenizer}


def indexer(texts: StrSequence,
            tokenize: Optional[Tokenizer] = None,
            ignore_rarer_than: Optional[int] = None,
            ignore_in: Optional[StrSequence] = None,
            **kwargs: Any) -> Indexer:
    """Create an indexer whose vocab is based on the given documents.

    Keyword arguments are passed to Indexer constructor.

    Args:
        texts (StrSequence): Texts to extract vocab from.
        tokenize (Optional[Tokenizer], optional): Tokenizer to use to
            determine vocabulary. Defaults to `Tokenizer.default()`.
        ignore_rarer_than (Optional[int], optional): Forwarded to
            `Vocab.create`. Defaults to None.
        ignore_in (Optional[StrSequence]): Ignore all words in this list.
            May be case/lemma sensitive depending on tokeinzer settings.

    Returns:
        Indexer: The created indexer.

    """
    if tokenize is None:
        tokenize = tokenizer()
    vocabulary = vocab(texts,
                       tokenize=tokenize,
                       ignore_rarer_than=ignore_rarer_than,
                       ignore_in=ignore_in)
    return Indexer(vocabulary, tokenize, **kwargs)


def join(texts: Any, delimiter: str = ' ') -> str:
    """Check if the texts are joinable.

    Args:
        texts (Any): A string or iterable of strings. Anything else, and
            this function will explode!
        delimiter (str, optional): If texts is iterable of strings, join
            them with this token. Defaults to ' '.

    Returns:
        str: The joined string, if possible.

    """
    if isinstance(texts, (set, frozenset)):
        texts = tuple(sorted(texts))
    if isinstance(texts, (list, tuple)):
        texts = delimiter.join(texts)
    if not isinstance(texts, str):
        raise ValueError(f'unknown annotation type: {type(texts).__name__}')
    return texts
