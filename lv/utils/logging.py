"""Logging utilities."""
import collections
import random
from typing import Any, Callable, Optional, Sequence, Sized, Union, cast

from lv import datasets
from lv.utils.typing import StrSequence

import wandb
from PIL import Image
from torch.utils import data


def wandb_image(image: Image.Image, caption: str,
                **kwargs: Any) -> wandb.Image:
    """Create a wandb image.

    Keyword arguments are treated as metadata and are prepended to the caption.

    Args:
        image (Image.Image): The PIL image to show.
        caption (str): Caption for the image.

    Returns:
        wandb.Image: The wandb image.

    """
    metadata = ', '.join(f'{k}={v}' for k, v in kwargs.items())
    return wandb.Image(image, caption=f'({metadata}) {caption}')


def wandb_images(images: Sequence[Image.Image], captions: StrSequence,
                 **kwargs: Any) -> Sequence[wandb.Image]:
    """Convert all (image, caption) pairs into wandb images.

    Keyword arguments are collapsed into a string and prepended to each
    of the image captions. If a kwarg's value is a function, it will be
    called for each sample and passed the sample index as input.

    Args:
        images (Sequence[Image.Image]): The images to display.
        captions (StrSequence): The caption for each image.

    Raises:
        ValueError: If number of images and captions is different.

    Returns:
        Sequence[wandb.Image]: The wandb images.

    """
    if len(images) != len(captions):
        raise ValueError(f'got {len(images)} images, {len(captions)} captions')

    wandb_images = []
    for index, (image, caption) in enumerate(zip(images, captions)):
        metadata = collections.OrderedDict()
        for key, value in kwargs.items():
            metadata[key] = value(index) if callable(value) else value
        wandb_images.append(wandb_image(image, caption, **metadata))
    return tuple(wandb_images)


def random_wandb_images(
    dataset: data.Dataset,
    sample_to_caption: Callable[[int, Any], str],
    sample_to_image: Callable[[int, Any], Image.Image],
    indices: Optional[Sequence[int]] = None,
    k: int = 25,
    **kwargs: Any,
) -> Sequence[wandb.Image]:
    """Choose random samples from the dataset and map them to wandb images.

    Keyword arguments are collapsed into a string and prepended to each
    of the image captions. If a kwarg's value is a function, it will be
    called for each sample and passed both the index and the sample itself.

    Args:
        dataset (data.Dataset): The dataset.
        sample_to_caption (Callable[[int, Any], str]): Function that maps
            dataset index and sample to caption. Defaults to None.
        sample_to_image (Callable[[int, Any], Image.Image]): Function that maps
            dataset index and sample to a PIL image. By default, tries to
            call `sample.as_pil_image_grid()` to be compatible with
            TopImagesDataset and its variants.
        indices (Optional[Sequence[int]], optional): Indices to sample from.
            By default, uses all indices.

        k (int, optional): Number of samples. Defaults to 25.

    Returns:
        Sequence[wandb.Image]: The wandb images.

    """
    if indices is None:
        size = len(cast(Sized, dataset))
        indices = list(range(size))

    chosen = random.sample(indices, k=min(k, len(indices)))
    images = [sample_to_image(index, dataset[index]) for index in chosen]
    captions = [sample_to_caption(index, dataset[index]) for index in chosen]

    metadata = collections.OrderedDict()
    for key, value in kwargs.items():
        if callable(value):
            metadata[key] = lambda index: value(index, dataset[index])
        else:
            metadata[key] = value

    return wandb_images(images, captions, **metadata)


AnyTopImagesDataset = Union[datasets.TopImagesDataset,
                            datasets.AnnotatedTopImagesDataset]


def random_neuron_wandb_images(dataset: AnyTopImagesDataset,
                               captions: StrSequence,
                               indices: Optional[Sequence[int]] = None,
                               k: int = 25,
                               **kwargs: Any) -> Sequence[wandb.Image]:
    """Sample neurons and convert their top image grids to wandb images.

    Automatically include layer/unit metadata with each image unless it is
    overridden by the caller.

    Args:
        dataset (AnyTopImagesDataset): The dataset to sample from.
        captions (Sequence[str]): Captions for the top images.
        indices (Optional[Sequence[int]], optional): Indices to sample from.
            By default, uses all indices.
        k (int, optional): Number of samples to draw. Defaults to 25.

    Returns:
        Sequence[wandb.Image]: The wandb images.

    """
    kwargs.setdefault('layer', lambda _, sample: sample.layer)
    kwargs.setdefault('unit', lambda _, sample: sample.unit)
    return random_wandb_images(dataset,
                               lambda _, sample: sample.as_pil_image_grid(),
                               lambda i, _: captions[i],
                               indices=indices,
                               k=k,
                               **kwargs)
