"""Common transforms for input, hidden, and output data during dissection."""
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar

from lv.utils.typing import Device

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
