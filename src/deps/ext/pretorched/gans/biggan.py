"""Extensions for pretorched/gans/biggan module."""
import collections
from typing import Any, List, NamedTuple, Sequence, Tuple

from src.deps.pretorched import layers
from src.deps.pretorched.gans import biggan

import torch
from torch import nn


def __getattr__(name: str) -> Any:
    """Forward to `lv.deps.pretorched.gans.biggan`."""
    return getattr(biggan, name)


class GInputs(NamedTuple):
    """Wraps inputs for sequential BigGAN."""

    # Follow naming conventions from pretorched.
    z: torch.Tensor
    y: torch.Tensor


class GBlockDataBag(NamedTuple):
    """Wraps outputs of a single GBlock."""

    # Follow naming conventions from pretorched.
    h: torch.Tensor
    ys: Sequence[torch.Tensor]


class SeqGPreprocess(nn.Module):
    """Module that wraps all preprocessing steps of BigGAN."""

    def __init__(self, generator: biggan.Generator):
        """Wrap the preprocessing steps of the given generator.

        Args:
            generator (biggan.Generator): The BigGAN generator.

        """
        super().__init__()
        self.generator = generator

    def forward(self, inputs: GInputs) -> GBlockDataBag:
        """Perform initialization steps for BigGAN generator.

        This includes preprocessing z/ys before handing them to GBlock modules.
        Mostly just copied from pretorched.

        Args:
            inputs (GInputs): Generator inputs, i.e. z and y, plus
                some other options.

        Returns:
            GBlockDataBag: Inputs for the first GBlock.

        """
        z, y = inputs

        # Here's the only difference between this and the original code:
        # We always embed.
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


class SeqGOutput(nn.Module):
    """Wraps the output layer of BigGAN."""

    def __init__(self, generator: biggan.Generator):
        """Wrap the given BigGAN generator.

        Args:
            generator (biggan.Generator): BigGAN generator.

        """
        super().__init__()
        self.generator = generator

    def forward(self, inputs: GBlockDataBag) -> torch.Tensor:
        """Transform the block outputs into images.

        Args:
            inputs (GBlockDataBag): Outputs from the last GBlock.

        Returns:
            torch.Tensor: The generated images.

        """
        return torch.tanh(self.generator.output_layer(inputs.h))


def SeqBigGAN(*args: Any, **kwargs: Any) -> nn.Sequential:
    """Return BigGAN as a sequential."""
    generator = biggan.BigGAN(*args, **kwargs)

    modules: List[Tuple[str, nn.Module]] = [
        ('preprocess', SeqGPreprocess(generator)),
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
    modules.append(('output', SeqGOutput(generator)))

    return nn.Sequential(collections.OrderedDict(modules))
