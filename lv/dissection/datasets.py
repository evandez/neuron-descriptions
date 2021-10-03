"""Custom datasets to be used in dissection."""
import pathlib
from typing import Any

from lv.utils.typing import PathLike

import torch
from torch.utils import data


class TensorDatasetOnDisk(data.TensorDataset):
    """Like `torch.utils.data.TensorDataset`, but tensors are pickled."""

    def __init__(self, root: PathLike, **kwargs: Any):
        """Load tensors from path and pass to `TensorDataset`.

        Args:
            root (PathLike): Root directory containing one or more .pth files
                of tensors.

        """
        loaded = []
        for child in pathlib.Path(root).iterdir():
            if not child.is_file() or not child.suffix == '.pth':
                continue
            tensors = torch.load(child, **kwargs)
            loaded.append(tensors)
        super().__init__(*loaded)
