"""Some useful type aliases relevant to this project."""
import pathlib
from typing import AbstractSet, List, Tuple, Union

import torch

Layer = Union[int, str]

PathLike = Union[str, pathlib.Path]

TensorPair = Tuple[torch.Tensor, torch.Tensor]
TensorTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

Device = Union[str, torch.device]

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = Union[List[str], Tuple[str, ...]]
StrIterable = Union[AbstractSet[str], StrSequence]
