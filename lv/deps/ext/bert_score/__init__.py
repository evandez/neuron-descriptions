"""Extensions for `bert_score` library."""
import os
from typing import Any

import bert_score
from transformers import logging


def __getattr__(name: str) -> Any:
    return getattr(bert_score, name)


class BERTScorer(bert_score.BERTScorer):
    """Wraps `bert_score.BERTScorer`, but silences useless warnings."""

    def __init__(self,
                 *args: Any,
                 silence_warnings: bool = True,
                 disable_parallelism: bool = True,
                 **kwargs: Any):
        """Initialize BERTScorer.

        This just forwards to `bert_score.BERTScorer`, but silences the
        pervasive warning that "not all weights have been initialized."

        Args:
            silence_warnings (bool, optional): Silence all `transformers`
                warnings when instantiating. Defaults to True.
            disable_parallelism (bool, optional): Disable `transformers`
                parallelism because it causes many warnings.

        """
        verbosity = logging.get_verbosity()
        if silence_warnings:
            logging.set_verbosity_error()

        if disable_parallelism:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        super().__init__(*args, **kwargs)

        if silence_warnings:
            logging.set_verbosity(verbosity)
