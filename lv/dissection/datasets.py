"""Custom datasets to be used in dissection."""
from typing import Any

from lv.utils.typing import PathLike

import torch
from torch.utils import data


class TensorDatasetOnDisk(data.TensorDataset):
    """Like `torch.utils.data.TensorDataset`, but tensors are pickled."""

    def __init__(self, path: PathLike, **kwargs: Any):
        """Load tensors from path and pass to `TensorDataset`.

        Args:
            path (PathLike): Path to file containing tensors.

        """
        tensors = torch.load(path, **kwargs)
        assert isinstance(tensors, tuple)
        super().__init__(*tensors)
