"""Pretrained word vectors used by other models."""
import warnings
from typing import Optional, cast

from lv.utils import lang

import spacy as sp
import torch
from spacy.lang import en
from torch import nn
from tqdm.auto import tqdm


def spacy(indexer: lang.Indexer,
          nlp: Optional[en.English] = None,
          display_progress: bool = True) -> nn.Embedding:
    """Return spacy embeddings for words in the given indexer's vocab.

    Args:
        indexer (lang.Indexer): Indexer containing vocab. Special tokens will
            be assigned zero vectors.
        nlp (Optional[en.English], optional): NLP object to use to get
            vectors. Defaults to new instance of en_core_web_lg.
        display_progress (bool, optional): Display progress bar when loading
            word vectors. Defaults to True.

    Raises:
        ValueError: If `nlp` has no pretrained vectors.

    Returns:
        nn.Embedding: The pretrained spacy embeddings.

    """
    if nlp is None:
        nlp = cast(en.English, sp.load('en_core_web_lg'))

    if not len(nlp.vocab.vectors):
        raise ValueError('found no vectors; you surely do not want this')

    words = tuple(enumerate(indexer.vocab.tokens))
    if display_progress:
        words = tqdm(words, desc='load spacy vectors')

    vectors = torch.zeros(len(indexer), nlp.vocab.vectors.shape[-1])
    unknown = []
    for index, word in words:
        vector = nlp.vocab.get_vector(word)
        if not vector.shape:
            unknown.append(word)
            continue
        vectors[index] = torch.from_numpy(vector)

    if unknown:
        warnings.warn(f'no embeddings for: {unknown}')

    return nn.Embedding.from_pretrained(vectors, padding_idx=indexer.pad_index)
