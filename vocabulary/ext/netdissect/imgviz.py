"""Extensions to the imgviz module."""
from typing import Any, Callable, Iterator, Sequence, Tuple, Union

from third_party.netdissect import imgviz, runningstats, tally
from vocabulary.typing import TensorPair, TensorTriplet

import torch
from PIL import Image
from torch.utils import data


def __getattr__(name: str) -> Any:
    return getattr(imgviz, name)


ComputeActivationsFn = Callable[..., Union[torch.Tensor, TensorPair]]
UnitRank = Tuple[int, int]
IndividualMaskedImages = Sequence[Sequence[Image.Image]]


class ImageVisualizer(imgviz.ImageVisualizer):
    """A NetDissect ImageVisualizer with additional utility functions."""

    def image_and_mask_grid_for_topk(self, compute: ComputeActivationsFn,
                                     dataset: data.Dataset,
                                     topk: runningstats.RunningTopK,
                                     **kwargs: Any) -> TensorTriplet:
        """Return top masks, images, and masked images.

        You can think of this method as a counterpart to the NetDissect
        `ImageVisualizer.masked_image_grid_for_topk` method. The only real
        difference is that, in addition to returning the masked images,
        it also returns the binary masks and the original unmasked images.

        The **kwargs are forwarded to `tally.gather_topk`.

        Args:
            compute (ComputeActivationsFn): Compute function. Should return
                activations given a batch from the dataset.
            dataset (data.Dataset): The dataset to compute activations on.
                See `tally.gather_topk`.
            topk (runningstats.RunningTopK): Top-k results.
                See `tally.gather_topk`.

        Returns:
            TensorTriplet: Image masks
                of shape (units, k, 1, height, width), unmasked images
                of shape (units, k, 3, height, width), masked images
                of shape (units, k, 3, height, width), all byte tensors.

        """

        def compute_viz(
                gather: Sequence[Sequence[UnitRank]],
                *batch: Any) -> Iterator[Tuple[UnitRank, torch.Tensor]]:
            activations = compute(*batch)
            if isinstance(activations, tuple):
                activations, images = activations
            else:
                images, *_ = batch
            for ranks, acts, image in zip(gather, activations, images):
                for unit, rank in ranks:
                    image = self.pytorch_image(image).cpu()
                    mask = self.pytorch_mask(acts, unit).cpu()
                    masked = self.pytorch_masked_image(image,
                                                       mask=mask,
                                                       outside_bright=.25,
                                                       thickness=0).cpu()
                    result = torch.cat([
                        component.float().clamp(0, 255).byte()
                        for component in (masked, image, mask[None])
                    ])
                    yield ((unit, rank), result)

        gt = tally.gather_topk(compute_viz, dataset, topk, **kwargs).result()
        masked, images, mask = gt[:, :, :3], gt[:, :, 3:6], gt[:, :, 6:]
        return masked, images, mask

    def individual_masked_images_for_topk(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Sequence[Sequence[Image.Image]], torch.Tensor, torch.Tensor]:
        """Return individual masked PIL images and separate masks/images.

        You can think of this method as a counterpart to the NetDissect
        `ImageVisualizer.individual_masked_images_for_topk` method. The only
        difference is it also returns the masks and images separately.
        """
        grids = self.image_and_mask_grid_for_topk(*args, **kwargs)
        masked, images, masks = grids
        masked = masked.permute(0, 1, 3, 4, 2)
        individual = [
            [Image.fromarray(mi.numpy()) for mi in mis] for mis in masked
        ]
        return individual, images, masks
