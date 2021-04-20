"""Utilities for preprocessing language data."""
import collections
import dataclasses
import functools
from typing import Any, Mapping, Optional, Sequence, Union, overload

from lv.utils.typing import StrIterable, StrSequence, StrSet

import spacy
from spacy.lang import en


@dataclasses.dataclass(frozen=True)
class Tokenizer:
    """A wrapper around the spacy English tokenizer supporting defaults."""

    nlp: en.English
    lemmatize: bool = True
    lowercase: bool = True
    ignore_stop: bool = True
    ignore_punct: bool = True

    @overload
    def __call__(self, texts: str) -> StrSequence:
        ...

    @overload
    def __call__(self, texts: StrSequence) -> Sequence[StrSequence]:
        ...

    def __call__(self, texts):
        """Tokenize the given texts.

        Args:
            texts (str or StrSequence): One or more text strings.
                See overloads.

        Returns:
            StrSequence or Sequence[StrSequence]: Tokenized strings.
                See overloads.

        """
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
                tokens.append(text)
            tokenized.append(tuple(tokens))

        if isinstance(texts, str):
            tokenized, = tokenized
        else:
            tokenized = tuple(tokenized)

        return tokenized


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
        nlp = spacy.load('en_core_web_sm')
        nlp.select_pipes(disable=('tok2vec', 'ner'))
        if not lemmatize:
            nlp.disable_pipe('lemmatizer')
    return Tokenizer(nlp, lemmatize=lemmatize, **kwargs)


@dataclasses.dataclass(frozen=True)
class Vocab:
    """A data class that stores tokens and a corresponding tokenizer."""

    tokens: Sequence[str]

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

    def __getitem__(self, token):
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

    def ignore(token, count):
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


@dataclasses.dataclass(frozen=True)
class Indexer:
    """Maps string text to integer ID sequences."""

    vocab: Vocab
    tokenize: Tokenizer
    start: bool = False
    stop: bool = False
    pad: bool = False
    unk: bool = False
    length: Optional[int] = None

    @property
    def start_index(self) -> int:
        """Return the index of a (hypothetical) start token."""
        return len(self.vocab)

    @property
    def stop_index(self) -> int:
        """Return the index of a (hypothetical) stop token."""
        return len(self.vocab) + 1

    @property
    def pad_index(self) -> int:
        """Return index of a (hypothetical) padding token."""
        return len(self.vocab) + 2

    @property
    def unk_index(self) -> int:
        """Return index of a (hypothetical) unknown token."""
        return len(self.vocab) + 3

    @overload
    def __call__(self,
                 texts: str,
                 start: Optional[bool] = ...,
                 stop: Optional[bool] = ...,
                 pad: Optional[bool] = ...,
                 unk: Optional[bool] = ...,
                 length: Optional[int] = ...) -> Sequence[int]:
        ...

    @overload
    def __call__(self,
                 texts: StrSequence,
                 start: Optional[bool] = ...,
                 stop: Optional[bool] = ...,
                 pad: Optional[bool] = ...,
                 unk: Optional[bool] = ...,
                 length: Optional[int] = ...) -> Sequence[Sequence[int]]:
        ...

    def __call__(self,
                 texts,
                 start=None,
                 stop=None,
                 pad=None,
                 unk=None,
                 length=None):
        """Tokenizer the given texsts and map them to integer IDs.

        Args:
            texts (str or StrSequence): One or more texts.
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
                length. Defaults to `self.length`, or if that is not set,
                defaults to None.

        Returns:
            Sequence[int] or Sequence[Sequence[int]]: The indexed sequence(s).

        """
        tokenized = self.tokenize([texts] if isinstance(texts, str) else texts)

        start = self.start if start is None else start
        stop = self.stop if stop is None else stop
        pad = self.pad if pad is None else pad
        unk = self.unk if unk is None else unk
        length = length or self.length or max(len(toks) for toks in tokenized)

        indexed = []
        for tokens in tokenized:
            indices = []

            if start:
                indices.append(self.start_index)

            if unk:
                indices += [
                    self.vocab.ids.get(tok, self.unk_index) for tok in tokens
                ]
            else:
                indices += [self.vocab[tok] for tok in tokens if tok in tokens]

            if stop:
                if len(indices) >= length:
                    indices = indices[:length - 1]
                indices.append(self.stop_index)

            if len(indices) < length and pad:
                indices += [self.pad_index] * (length - len(indices))
            elif len(indices) > length:
                indices = indices[:length]

            indexed.append(tuple(indices))

        if isinstance(texts, str):
            indexed, = indexed
        else:
            indexed = tuple(indexed)

        return indexed


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
