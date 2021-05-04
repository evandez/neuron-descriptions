"""Functions for dissecting convolutional units in vision models."""
import pathlib
import shutil
from typing import Any, Callable, Optional

from lv.dissection import transforms
from lv.ext.netdissect import imgviz
from lv.utils.typing import Device, Layer, PathLike, TensorPair
from third_party.netdissect import imgsave, nethook, pbar, renormalize, tally

import numpy
import torch
from torch import nn
from torch.utils import data


def run(compute_topk_and_quantile: Callable[..., TensorPair],
        compute_activations: imgviz.ComputeActivationsFn,
        dataset: data.Dataset,
        k: int = 15,
        quantile: float = 0.99,
        output_size: int = 224,
        batch_size: int = 128,
        image_size: Optional[int] = None,
        renormalizer: Optional[renormalize.Renormalizer] = None,
        num_workers: int = 30,
        results_dir: Optional[PathLike] = None,
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
        compute_topk_and_quantile (Callable[..., TensorPair]): Function taking
            dataset batch as input and returning tuple of (1) pooled unit
            activations with shape (batch_size, units), and (2) unpooled unit
            activations with shape (*, units).
        compute_activations (imgviz.ComputeActivationsFn): Function taking
            dataset batch as input and returning activations with shape
            (batch_size, channels, *) and optionally the associated images
            of shape (batch_size, channels, height, width).
        dataset (data.Dataset): Dataset to compute activations on.
        k (int, optional): Number of top-activating images to save.
            Defaults to 15.
        quantile (float, optional): Activation quantile to use when visualizing
            top images. Defaults to 0.99 (top 1% of activations).
        batch_size (int, optional): Max number of images to send through the
            model at any given time. Defaults to 128.
        output_size (int, optional): Top images and masks will be resized to be
            square in this size. Defaults to 224.
        image_size (Optional[int], optional): Expected size of dataset images.
            If not set, will attempt to infer from the dataset's `transform`
            property. If dataset does not have `transform`, dissection fails.
            Defaults to None.
        renormalizer (Optional[renormalize.Renormalizer], optional): NetDissect
            renormalizer for the dataset images. If not set, NetDissect will
            attempt to infer it from the dataset's `transform` property.
            Defaults to None.
        num_workers (int, optional): When loading or saving data in parallel,
            use this many worker threads. Defaults to 30.
        results_dir (Optional[PathLike], optional): Directory to write
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

    Raises:
        ValueError: If `k` or `quantile` are invalid.

    """
    if k < 1:
        raise ValueError(f'must have k >= 1, got k={k}')
    if quantile <= 0 or quantile >= 1:
        raise ValueError('must have quantile in range (0, 1), '
                         f'got quantile={quantile}')
    if image_size is None and not hasattr(dataset, 'transform'):
        raise ValueError('dataset has no `transform` property so '
                         'image_size= must be set')
    if results_dir is None:
        results_dir = pathlib.Path(__file__).parents[2] / 'dissection-results'
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
        pbar.descnext('tally activations')
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
    viz = imgviz.ImageVisualizer(output_size,
                                 image_size=image_size,
                                 renormalizer=renormalizer,
                                 source=dataset,
                                 level=levels)
    masked, images, masks = viz.individual_masked_images_for_topk(
        compute_activations,
        dataset,
        topk,
        k=k,
        batch_size=batch_size,
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
    lightbox_dir = pathlib.Path(__file__).parents[2] / 'third_party'
    lightbox_file = lightbox_dir / 'lightbox.html'
    for unit in range(len(images)):
        unit_dir = results_dir / f'viz/unit_{unit}'
        unit_lightbox_file = unit_dir / '+lightbox.html'
        shutil.copy(lightbox_file, unit_lightbox_file)


def discriminative(
        model: nn.Module,
        dataset: data.Dataset,
        device: Optional[Device] = None,
        results_dir: Optional[PathLike] = None,
        transform_inputs: transforms.TransformToTuple = transforms.first,
        transform_outputs: transforms.TransformToTensor = transforms.identity,
        **kwargs: Any) -> None:
    """Run dissection on a discriminative model.

    That is, a model for which image goes in, prediction comes out. Its outputs
    will be interpretted as the neuron activations to track.

    Keyword arguments are forwarded to `run`.

    Args:
        model (nn.Module): The model to dissect.
        dataset (data.Dataset): Dataset of images used to compute the
            top-activating images.
        device (Optional[Device], optional): Run all computations on this
            device. Defaults to None.
        results_dir (PathLike, optional): Directory to write results to.
            Defaults to same as `run`.
        transform_inputs (transforms.TransformToTuple, optional): Pass batch
            as *args to this function and use output as *args to model.
            Defaults to identity, i.e. entire batch is passed to model.
        transform_outputs (transforms.TransformToTensor, optional): Pass output
            of entire model, i.e. the activations, to this function and hand
            result to netdissect. Defaults to identity function, i.e. the raw
            data will be tracked by netdissect.

    """
    model.to(device)

    def compute_topk_and_quantile(*inputs: Any) -> TensorPair:
        inputs = transforms.map_location(transform_inputs(*inputs), device)
        with torch.no_grad():
            outputs = model(*inputs)
        outputs = transform_outputs(outputs)
        batch_size, channels, *_ = outputs.shape
        activations = outputs.permute(0, 2, 3, 1).reshape(-1, channels)
        pooled, _ = outputs.view(batch_size, channels, -1).max(dim=2)
        return pooled, activations

    def compute_activations(*inputs: Any) -> torch.Tensor:
        inputs = transform_inputs(*inputs)
        with torch.no_grad():
            outputs = model(*inputs)
        outputs = transform_outputs(outputs)
        return outputs

    run(compute_topk_and_quantile,
        compute_activations,
        dataset,
        results_dir=results_dir,
        **kwargs)


def sequential(model: nn.Sequential,
               dataset: data.Dataset,
               layer: Optional[Layer] = None,
               results_dir: Optional[PathLike] = None,
               **kwargs: Any) -> None:
    """Run dissection on a sequential discriminative model.

    That is, a model for which image goes in, prediction comes out.
    Because this function assumes the model is a `torch.nn.Sequential`,
    you can specify the layer to look at activations for.

    Keyword arguments are forwarded to `discriminative`.

    Args:
        model (nn.Sequential): The sequential model to dissect.
        dataset (data.Dataset): Dataset of images used to compute the
            top-activating images.
        layer (Optional[Layer], optional): Track unit activations for this
            layer. If not set, NetDissect will only look at the final output
            of the model. Defaults to None.
        results_dir (PathLike, optional): Directory to write results to.
            If set and layer is also set, layer name will be appended to path.
            Defaults to same as `run`.

    """
    if results_dir is not None:
        results_dir = pathlib.Path(results_dir)
        results_dir /= str(layer) if layer is not None else 'outputs'
    if layer is not None:
        model = nethook.subsequence(model,
                                    last_layer=layer,
                                    share_weights=True)
    discriminative(model, dataset, results_dir=results_dir, **kwargs)


def generative(
        model: nn.Sequential,
        dataset: data.Dataset,
        layer: Layer,
        device: Optional[Device] = None,
        results_dir: Optional[PathLike] = None,
        transform_inputs: transforms.TransformToTuple = transforms.identities,
        transform_hiddens: transforms.TransformToTensor = transforms.identity,
        transform_outputs: transforms.TransformToTensor = transforms.identity,
        **kwargs: Any) -> None:
    """Run dissection on a generative model of images.

    That is, a model for which representation goes in, image comes out.
    Because of the way these models are structured, we need both the generated
    images and the intermediate activation. To facilitate this, we require the
    model be implemented as a `torch.nn.Sequential` so we can slice it up and
    look at intermediate values while also saving the generated images for
    visualization downstream.

    Keyword arguments are forwarded to `run`.

    Args:
        model (nn.Sequential): The model to dissect.
        dataset (data.Dataset): Dataset of representations used to generate
            images. The top-activating images will be taken from them.
        layer (Layer): Track unit activations for this layer.
        device (Optional[Device], optional): Run all computations on this
            device. Defaults to None.
        results_dir (PathLike, optional): Directory to write results to.
            If set, layer name will be appended to path. Defaults to same
            as `run`.
        transform_inputs (transforms.TransformToTuple, optional): Pass batch
            as *args to this function and use output as *args to model.
            Defaults to identity, i.e. entire batch is passed to model.
        transform_hiddens (transforms.TransformToTensor, optional): Pass output
            of intermediate layer to this function and hand the result to
            netdissect. This is useful if e.g. your model passes info between
            layers as a dictionary or other non-Tensor data type.
            Defaults to the identity function, i.e. the raw data will be
            tracked by netdissect.
        transform_outputs (transforms.TransformToTensor, optional): Pass output
            of entire model, i.e. generated images, to this function and hand
            result to netdissect. Defaults to identity function, i.e. the raw
            data will be tracked by netdissect.

    """
    if results_dir is not None:
        results_dir = pathlib.Path(results_dir) / str(layer)

    model.to(device)
    with nethook.InstrumentedModel(model) as instrumented:
        instrumented.retain_layer(layer)

        def compute_topk_and_quantile(*inputs: Any) -> TensorPair:
            inputs = transforms.map_location(transform_inputs(*inputs), device)
            with torch.no_grad():
                model(*inputs)
            hiddens = transform_hiddens(instrumented.retained_layer(layer))
            batch_size, channels, *_ = hiddens.shape
            activations = hiddens.permute(0, 2, 3, 1).reshape(-1, channels)
            pooled, _ = hiddens.view(batch_size, channels, -1).max(dim=2)
            return pooled, activations

        def compute_activations(*inputs: Any) -> TensorPair:
            inputs = transform_inputs(*inputs)
            with torch.no_grad():
                images = model(*inputs)
            hiddens = transform_hiddens(instrumented.retained_layer(layer))
            images = transform_outputs(images)
            return hiddens, images

        run(compute_topk_and_quantile,
            compute_activations,
            dataset,
            results_dir=results_dir,
            **kwargs)
