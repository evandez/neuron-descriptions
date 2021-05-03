"""Extensions for pretorched/gans/biggan module."""
import collections
import dataclasses
from typing import Any, List, Sequence, Tuple

from third_party.pretorched.gans import biggan
from third_party.pretorched import layers

import torch
from torch import nn


def __getattr__(name):
    """Forward to `pretorched.gans.biggan`."""
    return getattr(biggan, name)


@dataclasses.dataclass(frozen=True)
class GInitDataBag:
    """Wraps inputs for SeqGInit."""

    # Follow naming conventions from pretorched.
    z: torch.Tensor
    y: torch.Tensor
    embed: bool = True


@dataclasses.dataclass(frozen=True)
class GBlockDataBag:
    """Wraps outputs of a single GBlock."""

    # Follow naming conventions from pretorched.
    h: torch.Tensor
    ys: Sequence[torch.Tensor]


class SeqGInit(nn.Module):
    """Module that wraps all initialization steps of BigGAN."""

    def __init__(self, generator: biggan.Generator):
        """Wrap the preprocessing steps of the given generator.

        Args:
            generator (biggan.Generator): The BigGAN generator.

        """
        super().__init__()
        self.generator = generator

    def forward(self, inputs: GInitDataBag) -> GBlockDataBag:
        """Perform initialization steps for BigGAN generator.

        This includes preprocessing z/ys before handing them to GBlock modules.
        Mostly just copied from pretorched.

        Args:
            inputs (GInitDataBag): Generator inputs, i.e. z and y, plus some
                other options.

        Returns:
            GBlockDataBag: Inputs for the first GBlock.

        """
        z, y, embed = inputs.z, inputs.y, inputs.embed
        if embed:
            if y.ndim > 1:
                y = y @ self.generator.shared.weight
            else:
                y = self.generator.shared(y)

        # If hierarchical, concatenate zs and ys
        if self.generator.hier:
            zs = torch.split(z, self.generator.z_chunk_size, 1)
            z = zs[0]
            ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            ys = [y] * len(self.generator.blocks)

        # First linear layer
        h = self.generator.linear(z)

        # Reshape
        h = h.view(h.size(0), -1, self.generator.bottom_width,
                   self.generator.bottom_width)

        return GBlockDataBag(h, ys)


class SeqGBlock(nn.Module):
    """A GBlock that can be used with nn.Sequential."""

    def __init__(self, block: biggan.GBlock, index: int):
        """Wrap the given block.

        Args:
            block (biggan.GBlock): The GBlock to wrap.
            index (int): Index of the GBlock among all other blocks.

        """
        super().__init__()
        self.block = block
        self.index = index

    def forward(self, inputs: GBlockDataBag) -> GBlockDataBag:
        """Pass the inputs to the block and carry on."""
        h, ys = inputs.h, inputs.ys
        h = self.block(h, ys[self.index])
        return GBlockDataBag(h, ys)


def SeqBigGAN(*args: Any, **kwargs: Any) -> nn.Sequential:
    """Return BigGAN as a sequential."""
    generator = biggan.BigGAN(*args, **kwargs)

    modules: List[Tuple[str, nn.Module]] = [
        ('preprocess', SeqGInit(generator)),
    ]
    for index, blocks in enumerate(generator.blocks):
        assert len(blocks) <= 2, 'should never be more than 2 blocks'
        for block in blocks:
            if isinstance(block, biggan.GBlock):
                key = 'layer'
            else:
                assert isinstance(block, layers.Attention), 'unknown block'
                key = 'attn'
            key += str(index)
            modules.append((key, SeqGBlock(block, index)))

    return nn.Sequential(collections.OrderedDict(modules))
