"""Ablation utilities."""
import collections
import contextlib
from typing import Callable, ContextManager, Sequence

from lv.third_party.netdissect import nethook
from lv.utils.typing import Unit

import torch
from torch import nn

Rule = Callable[[torch.Tensor], torch.Tensor]
RuleFactory = Callable[[Sequence[int]], Rule]


def zero(units: Sequence[int]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Zero the given units.

    Args:
        units (Sequence[int]): The units to zero.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: Function that takes layer
            features and zeros the given units, returning the result.

    """

    def fn(features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 4:
            raise ValueError(f'expected 4D features, got {features.dim()}')
        # Make sure we don't break autograd by editing values in place.
        # Just use a mask. Fauci said it first.
        shape = (*features.shape[:2], 1, 1)
        mask = features.new_ones(*shape)
        mask[:, units] = 0
        return features * mask

    return fn


def ablated(
    model: nn.Sequential,
    units: Sequence[Unit],
    rule: RuleFactory = zero,
) -> ContextManager[nethook.InstrumentedModel]:
    """Ablate the given units according to the given rule.

    Args:
        model (nn.Sequential): The model to ablate.
        units (Sequence[Unit]): The (layer, unit) pairs to ablate.
        rule (RuleFactory, optional): The rule to ablate to. Defaults to
            zeroing the units.

    Yields:
        ContextManager[nn.Sequential]: An InstrumentedModel configured to
            ablate the units.

    """
    # Dirty hack to get type checking to work.
    @contextlib.contextmanager
    def wrapper():
        with nethook.InstrumentedModel(model) as instrumented:
            edits = collections.defaultdict(list)
            for la, un in units:
                edits[la].append(un)
            for la, uns in edits.items():
                instrumented.edit_layer(la, rule=rule(sorted(uns)))
            yield instrumented

    return wrapper()
