"""Some useful type aliases relevant to this project."""
import pathlib
from typing import Tuple, Union

import torch

Layer = Union[int, str]

PathLike = Union[str, pathlib.Path]

TensorPair = Tuple[torch.Tensor, torch.Tensor]
TensorTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

Device = Union[str, torch.device]
