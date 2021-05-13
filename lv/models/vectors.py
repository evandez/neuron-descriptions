"""Pretrained word vectors used by other models."""
import warnings
from typing import Optional, cast

from lv.utils import lang

import spacy as sp
import torch
from spacy.lang import en
from torch import nn


def spacy(indexer: lang.Indexer,
          nlp: Optional[en.English] = None) -> nn.Embedding:
    """Return spacy embeddings for words in the given indexer's vocab.

    Args:
        indexer (lang.Indexer): Indexer containing vocab. Special tokens will
            be assigned zero vectors.
        nlp (Optional[en.English], optional): NLP object to use to get
            vectors. Defaults to new instance of en_core_web_lg.

    Raises:
        ValueError: If `nlp` has no pretrained vectors.

    Returns:
        nn.Embedding: The pretrained spacy embeddings.

    """
    if nlp is None:
        nlp = cast(en.English, sp.load('en_core_web_lg'))

    if not len(nlp.vocab.vectors):
        raise ValueError('found no vectors; you surely do not want this')

    vectors = torch.zeros(len(indexer), nlp.vocab.vectors.shape[-1])
    unknown = []
    for index, word in enumerate(indexer.vocab.tokens):
        vector = nlp.vocab.get_vector(word)
        if not vector.shape:
            unknown.append(word)
            continue
        vectors[index] = torch.from_numpy(vector)

    if unknown:
        warnings.warn(f'no embeddings for: {unknown}')

    return nn.Embedding.from_pretrained(vectors, padding_idx=indexer.pad_index)
