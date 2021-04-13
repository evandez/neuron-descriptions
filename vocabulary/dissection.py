"""Functions for dissecting individual units in vision models."""
import pathlib
import shutil
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple, Union

from third_party.netdissect import (imgsave, imgviz, nethook, pbar,
                                    runningstats, tally)

import numpy
import torch
from PIL import Image
from torch import nn
from torch.utils import data

Layer = Union[int, str]

TensorPair = Tuple[torch.Tensor, torch.Tensor]
ComputeTopKAndQuantileFn = Callable[..., TensorPair]
ComputeActivationsFn = Callable[..., Union[torch.Tensor, TensorPair]]

PathLike = Union[str, pathlib.Path]

UnitRank = Tuple[int, int]
IndividualMaskedImages = Sequence[Sequence[Image.Image]]


class ImageVisualizer(imgviz.ImageVisualizer):
    """A NetDissect ImageVisualizer with additional utility functions."""

    def image_and_mask_grid_for_topk(
            self, compute: ComputeActivationsFn, dataset: data.Dataset,
            topk: runningstats.RunningTopK,
            **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Image masks
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
                    image = self.pytorch_image(image)
                    mask = self.pytorch_mask(acts, unit)
                    masked = self.pytorch_masked_image(image,
                                                       mask=mask,
                                                       outside_bright=.25,
                                                       thickness=0)
                    result = torch.cat([
                        component.clamp(0, 255).byte().cpu()
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
        individual = [
            [Image.fromarray(mi.numpy()) for mi in mis] for mis in masked
        ]
        return individual, images, masks


def dissect(compute_topk_and_quantile: ComputeTopKAndQuantileFn,
            compute_activations: ComputeActivationsFn,
            dataset: data.Dataset,
            k: int = 15,
            quantile: float = 0.99,
            batch_size: int = 128,
            image_size: int = 224,
            num_workers: int = 30,
            results_dir: PathLike = 'dissection-results',
            tally_cache_file: Optional[PathLike] = None,
            masks_cache_file: Optional[PathLike] = None,
            clear_cache_files: bool = False,
            clear_results_dir: bool = False,
            display_progress: bool = True) -> None:
    """Find and visualize the top-activating images for each unit.

    Top-activating images are found with network dissection [Bau et al., 2017].
    This function just forwards to the NetDissect library. We do not explicitly
    take a model as input but rather two blackbox functions which take dataset
    batches as input and return unit activations as output.

    Args:
        compute_topk_and_quantile (ComputeTopKAndQuantileFn): Function taking
            dataset batch as input and returning tuple of (1) pooled unit
            activations with shape (batch_size, units), and (2) unpooled unit
            activations with shape (*, units).
        compute_activations (ComputeActivationsFn): Function taking dataset
            batch as input and returning activations with shape
            (batch_size, channels, *) and optionally the associated images
            of shape (batch_size, channels, height, width).
        dataset (data.Dataset): Dataset to compute activations on.
        k (int, optional): Number of top-activating images to save.
            Defaults to 15.
        quantile (float, optional): Activation quantile to use when visualizing
            top images. Defaults to 0.99 (top 1% of activations).
        batch_size (int, optional): Max number of images to send through the
            model at any given time. Defaults to 128.
        image_size (int, optional): Top images will be resized to be square
            in this size. Defaults to 224.
        num_workers (int, optional): When loading or saving data in parallel,
            use this many worker threads. Defaults to 30.
        results_dir (PathLike, optional): Directory to write
            results to. Defaults to 'dissection-results'.
        tally_cache_file (Optional[PathLike], optional): Write intermediate
            results for tally step to this file. Defaults to None.
        masks_cache_file (Optional[PathLike], optional): Write intermediate
            results for determining top-k image masks to this file.
            Defaults to None.
        clear_cache_files (bool, optional): If set, clear existing cache files
            with the same name as any of the *_cache_file arguments to this
            function. Useful if you want to redo all computation.
            Defaults to False.
        clear_results_dir (bool, optional): If set, clear the results_dir if
            it exists. Defaults to False.
        display_progress (bool, optional): If True, display progress bar.
            Defaults to True.

    """
    if k < 1:
        raise ValueError(f'must have k >= 1, got k={k}')
    if quantile <= 0 or quantile >= 1:
        raise ValueError('must have quantile in range (0, 1), '
                         f'got quantile={quantile}')
    if not isinstance(results_dir, pathlib.Path):
        results_dir = pathlib.Path(results_dir)

    # Clear cache files if requested.
    if clear_cache_files:
        for cache_file in (tally_cache_file, masks_cache_file):
            if cache_file is None:
                continue
            cache_file = pathlib.Path(cache_file)
            if cache_file.exists():
                cache_file.unlink()

    # Clear results directory if requested.
    if clear_results_dir and results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    # Compute activation statistics across dataset.
    if display_progress:
        pbar.descnext('rq')
    topk, rq = tally.tally_topk_and_quantile(compute_topk_and_quantile,
                                             dataset,
                                             k=k,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             cachefile=tally_cache_file)

    # Now compute top images and masks for the highest-activating pixels.
    if display_progress:
        pbar.descnext('compute top images')
    levels = rq.quantiles(quantile).reshape(-1)
    viz = ImageVisualizer(image_size, source=dataset, level=levels)
    masked, images, masks = viz.individual_masked_images_for_topk(
        compute_activations,
        dataset,
        topk,
        k=k,
        num_workers=num_workers,
        pin_memory=True,
        cachefile=masks_cache_file)

    # Now save the top images with the masks overlaid. We save each image
    # individually so they can be visualized and/or shown on MTurk.
    if display_progress:
        pbar.descnext('saving top images')
    numpy.save(f'{results_dir}/images.npy', images)
    numpy.save(f'{results_dir}/masks.npy', masks)
    imgsave.save_image_set(masked,
                           f'{results_dir}/viz/unit_%d/image_%d.png',
                           sourcefile=masks_cache_file)

    # Finally, save the IDs of all top images and the activation values
    # associated with each unit and image.
    activations, ids = topk.result()
    for metadata, name, fmt in ((activations, 'activations', '%.5e'),
                                (ids, 'ids', '%i')):
        metadata = metadata.view(len(images), k).cpu().numpy()
        metadata_file = results_dir / f'{name}.csv'
        numpy.savetxt(str(metadata_file), metadata, delimiter=',', fmt=fmt)

    # The lightbox lets us view all the masked images at once. Pretty handy!
    lightbox_dir = pathlib.Path(__file__).parents[1] / 'third_party'
    lightbox_file = lightbox_dir / 'lightbox.html'
    for unit in range(len(images)):
        unit_dir = results_dir / f'viz/unit_{unit}'
        unit_lightbox_file = unit_dir / '+lightbox.html'
        shutil.copy(lightbox_file, unit_lightbox_file)


def discriminative(model: nn.Sequential,
                   dataset: data.Dataset,
                   layer: Optional[Layer] = None,
                   device: Optional[torch.device] = None,
                   **kwargs: Any) -> None:
    """Run dissection on a discriminative model.

    That is, a model for which image goes in, prediction comes out.

    Args:
        model (nn.Sequential): The model to dissect. It must be a
            `torch.nn.Sequential` so we can slice it up and look at
            the activations.
        dataset (data.Dataset): Dataset of images used to compute the
            top-activating images.
        layer (Optional[Layer], optional): Track unit activations for this
            layer. If not set, NetDissect will only look at the final output
            of the model. Defaults to None.
        device (Optional[torch.device], optional): Run all computations on this
            device. Defaults to None.

    """
    model.to(device)
    model = nethook.subsequence(model, last_layer=layer)

    def compute_topk_and_quantile(images: torch.Tensor, *_: Any) -> TensorPair:
        with torch.no_grad():
            outputs = model(images.to(device))
        batch_size, channels, *_ = outputs.shape
        activations = outputs.permute(0, 2, 3, 1).reshape(-1, channels)
        pooled, _ = outputs.view(batch_size, channels, -1).max(dim=2)
        return pooled, activations

    def compute_activations(images: torch.Tensor, *_: Any) -> torch.Tensor:
        with torch.no_grad():
            outputs = model(images.to(device))
        return outputs

    dissect(compute_topk_and_quantile, compute_activations, dataset, **kwargs)


def generative(model: nn.Sequential,
               dataset: data.Dataset,
               layer: Optional[Layer] = None,
               device: Optional[torch.device] = None,
               **kwargs: Any) -> None:
    """Run dissection on a generative model of images.

    That is, a model for which representation goes in, image comes out.

    Args:
        model (nn.Sequential): The model to dissect. It must be a
            `torch.nn.Sequential` so we can slice it up and look at
            the activations.
        dataset (data.Dataset): Dataset of representations used to generate
            images. The top-activating images will be taken from them.
        layer (Optional[Layer], optional): Track unit activations for this
            layer. If not set, NetDissect will only look at the final output
            of the model. Defaults to None.
        device (Optional[torch.device], optional): Run all computations on this
            device. Defaults to None.

    """
    model.to(device)

    with nethook.InstrumentedModel(model) as instrumented:
        instrumented.retain_layer(layer)

        def compute_topk_and_quantile(zs: torch.Tensor, *_: Any) -> TensorPair:
            with torch.no_grad():
                model(zs.to(device))
            output = instrumented.retained_layer(layer)
            batch_size, channels, *_ = output.shape
            activations = output.permute(0, 2, 3, 1).reshape(-1, channels)
            pooled = output.view(batch_size, channels, -1).max(dim=2)
            return pooled, activations

        def compute_activations(zs: torch.Tensor, *_: Any) -> TensorPair:
            with torch.no_grad():
                images = model(zs.to(device))
            return instrumented.retained_layer(layer), images

        dissect(compute_topk_and_quantile, compute_activations, dataset,
                **kwargs)
