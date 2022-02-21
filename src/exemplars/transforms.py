"""Transforms for inputs/hiddens/outputs during exemplar computation."""
import math
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar

from src.utils.typing import Device

import torch


def map_location(data: Sequence[Any],
                 device: Optional[Device]) -> Tuple[Any, ...]:
    """Map all tensors in the sequence to the given device.

    Args:
        data (Any): The data to map.
        device (Optional[Device]): Map all tensors to this device.

    Returns:
        Tuple[Any, ...]: The same sequence, but tensors are mapped to the
            given device.

    """
    mapped = []
    for item in data:
        if isinstance(item, torch.Tensor) and device is not None:
            item = item.to(device)
        mapped.append(item)
    return tuple(mapped)


Transform = Callable[..., Any]
TransformToTuple = Callable[..., Tuple[Any, ...]]
TransformToTensor = Callable[..., torch.Tensor]


def first(*inputs: Any) -> Tuple[Any, ...]:
    """Return the first argument as a tuple."""
    return (inputs[0],)


T = TypeVar('T')


def identity(inputs: T) -> T:
    """Return the inputs."""
    return inputs


def identities(*inputs: T) -> Tuple[T, ...]:
    """Return all inputs as a tuple."""
    return inputs


def spatialize_vit_mlp(hiddens: torch.Tensor) -> torch.Tensor:
    """Make ViT MLP activations look like convolutional activations.

    Each activation corresponds to an image patch, so we can arrange them
    spatially. This allows us to use all the same dissection tools we
    used for CNNs.

    Args:
        hiddens (torch.Tensor): The hidden activations. Should have shape
            (batch_size, n_patches, n_units).

    Returns:
        torch.Tensor: Spatially arranged activations, with shape
            (batch_size, n_units, sqrt(n_patches - 1), sqrt(n_patches - 1)).
    """
    batch_size, n_patches, n_units = hiddens.shape

    # Exclude CLS token.
    hiddens = hiddens[:, :-1]
    n_patches -= 1

    # Compute spatial size.
    size = math.isqrt(n_patches)
    assert size**2 == n_patches

    # Finally, reshape.
    return hiddens.permute(0, 2, 1).reshape(batch_size, n_units, size, size)
