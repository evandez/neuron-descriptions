"""Some useful type aliases relevant to this project."""
import pathlib
from typing import AbstractSet, Callable, List, Mapping, Optional, Tuple, Union

import torch

Layer = Union[int, str]
Unit = Tuple[Layer, int]

PathLike = Union[str, pathlib.Path]

TensorPair = Tuple[torch.Tensor, torch.Tensor]
TensorTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
OptionalTensors = Tuple[Optional[torch.Tensor], ...]
StateDict = Mapping[str, torch.Tensor]

Device = Union[str, torch.device]

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = Union[List[str], Tuple[str, ...]]
StrSet = AbstractSet[str]
StrIterable = Union[StrSet, StrSequence]
StrMapping = Mapping[str, str]

# Some common transforms.
TransformTensor = Callable[[torch.Tensor], torch.Tensor]
TransformStr = Callable[[str], str]
TransformStrSeq = Callable[[StrSequence], StrSequence]
