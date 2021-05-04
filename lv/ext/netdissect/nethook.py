"""Extensions to the nethook module."""
from typing import Any, Optional

from lv.utils.typing import Layer
from third_party.netdissect import nethook

from torch import nn


def __getattr__(name):
    """Forward to original nethook module."""
    return getattr(nethook, name)


def subsequence(sequential: nn.Sequential,
                first_layer: Optional[Layer] = None,
                last_layer: Optional[Layer] = None,
                after_layer: Optional[Layer] = None,
                upto_layer: Optional[Layer] = None,
                single_layer: Optional[Layer] = None,
                **kwargs: Any):
    """Like nethook.subsequence, but handle integer layer constraints."""
    constraints = (first_layer, last_layer, after_layer, upto_layer)
    if single_layer is not None:
        if any(constraints):
            raise ValueError('cannot set single_layer with other constraints')
        first_layer = single_layer
        last_layer = single_layer

    if first_layer is not None and after_layer is not None:
        raise ValueError('cannot set both first_layer and after_layer')
    if last_layer is not None and upto_layer is not None:
        raise ValueError('cannot set both last_layer and upto_layer')

    constraints = (first_layer, last_layer, after_layer, upto_layer)
    if all(isinstance(c, int) or c is None for c in constraints):
        if isinstance(last_layer, int):
            sequential = sequential[:last_layer + 1]
        elif isinstance(upto_layer, int):
            sequential = sequential[:upto_layer]

        if isinstance(first_layer, int):
            sequential = sequential[first_layer:]
        elif isinstance(after_layer, int):
            sequential = sequential[after_layer + 1:]
        return sequential

    return nethook.subsequence(sequential,
                               first_layer=first_layer,
                               last_layer=last_layer,
                               after_layer=after_layer,
                               upto_layer=upto_layer,
                               single_layer=single_layer,
                               **kwargs)
