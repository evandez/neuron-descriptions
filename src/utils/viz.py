"""Visualization utilities."""
import collections
import pathlib
import random
from typing import Any, Callable, Optional, Sequence, Sized, Tuple, Union, cast

from src.deps.netdissect import imgsave
from src.milannotations import datasets
from src.utils.typing import PathLike, StrMapping, StrSequence

import wandb
from PIL import Image
from torch.utils import data
from tqdm.auto import tqdm


def kwargs_to_str(**kwargs: Any) -> str:
    """Return metadata as a compact string."""
    kvs = []
    for key, value in kwargs.items():
        if isinstance(value, float):
            kv = f'{key}={value:.2f}'
        elif isinstance(value, str):
            kv = f'{key}="{value}"'
        else:
            kv = f'{key}={value}'
        kvs.append(kv)
    return ', '.join(kvs)


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
    metadata = kwargs_to_str(**kwargs)
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
    sample_to_image: Callable[[int, Any], Image.Image],
    sample_to_caption: Callable[[int, Any], str],
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
        sample_to_image (Callable[[int, Any], Image.Image]): Function that maps
            dataset index and sample to a PIL image.
        sample_to_caption (Callable[[int, Any], str]): Function that maps
            dataset index and sample to caption.
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


def random_neuron_wandb_images(dataset: data.Dataset[datasets.TopImages],
                               captions: StrSequence,
                               indices: Optional[Sequence[int]] = None,
                               k: int = 25,
                               **kwargs: Any) -> Sequence[wandb.Image]:
    """Sample neurons and convert their top image grids to wandb images.

    Automatically include layer/unit metadata with each image unless it is
    overridden by the caller.

    Args:
        dataset (data.Dataset[datasets.TopImages]): The dataset to sample from.
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


PredictedCaptions = Union[StrSequence, Sequence[StrMapping]]
GetHeaderFn = Callable[[datasets.AnyTopImages, int], str]
GetBaseUrlFn = Callable[[datasets.AnyTopImages, int], str]
GetUrlUnitIdFn = Callable[[datasets.AnyTopImages, int], int]


def generate_html(
    dataset: data.Dataset[datasets.AnyTopImages],
    out_dir: PathLike,
    predictions: Optional[PredictedCaptions] = None,
    get_header: Optional[GetHeaderFn] = None,
    get_base_url: Optional[GetBaseUrlFn] = None,
    get_unit_id: Optional[GetUrlUnitIdFn] = None,
    include_gt: bool = True,
    save_images: bool = True,
    grid_images: bool = False,
    image_size: Optional[Tuple[int, int]] = None,
) -> None:
    """Generate an HTML visualization of neuron top images and captions.

    Args:
        dataset (data.Dataset[AnyTopImages]): Dataset of neurons.
        out_dir (PathLike): Directory to write top images and final HTML file.
        predictions (Optional[PredictedCaptions]): Predicted captions to show
            for each neuron. Elements can be single strings (i.e., one
            prediction per neuron) or a mapping from labels (prediction kinds)
            to strings (predictions). Defaults to None.
        get_header (Optional[GetHeaderFn], optional): Fn returning the header
            text for each neuron. Arguments are top images and thier index in
            `dataset`. By default, header is "{layer}-{unit}".
        get_base_url (Optional[GetBaseUrlFn], optional): Function returning the
            base URL for a given sample. If this is not set, not images will
            be included in the generated HTML. Defaults to None.
        get_unit_id (Optional[GetUnitIdFn], optional): Function returning
            the unit ID to use in the URL for the top images. By default,
            its index in the dataset will be used. Note this does NOT affect
            how images are saved on disk! Defaults to None.
        include_gt (bool, optional): If set, also write ground truth
            captions to the HTML when possible. Defaults to True.
        save_images (bool, optional): If set, save top images in dir.
            Defaults to True.
        grid_images (bool, optional): If set, save all top images as a single
            grid instead of individually. Defaults to False.
        image_size (Optional[Tuple[int, int], optional): Height and width
            (in px) for each saved image. Defaults depend on whether
            grid_images is set or not.

    Raises:
        ValueError: If `captions` is set but has different length
            than `dataset`.

    """
    length = len(cast(Sized, dataset))
    if predictions is not None and len(predictions) != length:
        raise ValueError(f'expected {length} predictions, '
                         f'got {len(predictions)}')

    if image_size is None:
        image_height = 600 if grid_images else 224
        image_width = 1000 if grid_images else 224
    else:
        image_height, image_width = image_size

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    images = []
    if grid_images:
        image_file_name_pattern = 'top_images_%d.png'
    else:
        image_file_name_pattern = 'top_images_%d_%d.png'

    html = [
        '<!doctype html>',
        '<html>',
        '<head>',
        '<style>',
        'td { padding-right: 10px; }',
        '</style>',
        '</head>',
        '<body>',
    ]
    for index in tqdm(range(length), desc='compiling top images'):
        sample = dataset[index]
        key = f'{sample.layer}-{sample.unit}'

        if get_header is not None:
            header = get_header(sample, index)
        else:
            header = key

        base_url = None
        if get_base_url is not None:
            base_url = get_base_url(sample, index)

        if get_unit_id is None:
            unit_id = index
        else:
            unit_id = get_unit_id(sample, index)

        if base_url is None:
            image_urls = []
        elif grid_images:
            image_urls = [f'{base_url}/{image_file_name_pattern % unit_id}']
        else:
            image_urls = [
                f'{base_url}/{image_file_name_pattern % (unit_id, position)}'
                for position in range(len(sample.images))
            ]

        if save_images and grid_images:
            images.append(sample.as_pil_image_grid())
        elif save_images:
            images.append(sample.as_pil_images())

        html += [
            '<div>',
            f'<h2>{header}</h2>',
            '<div style="display: inline-block">',
        ]
        for image_url in image_urls:
            html += [
                f'<img src="{image_url}" alt="{key}" '
                f'style="height: {image_height}px; width: {image_width}px"'
                '/>'
            ]
        html += ['</div>']

        if include_gt and isinstance(sample, datasets.AnnotatedTopImages):
            html += ['<h3>human annotations</h3>', '<ul>']
            for annotation in sample.annotations:
                html += [f'<li>{annotation}</li>']
            html += ['</ul>']

        if predictions is not None:
            if include_gt:
                html += [
                    '<h3>predicted caption</h3>',
                ]
            prediction = predictions[index]
            if isinstance(prediction, str):
                html += ['<div>', prediction, '</div>']
            else:
                html += ['<table>']
                for label, caption in prediction.items():
                    html += [
                        '<tr>',
                        f'<td><b>{label}</b></td>',
                        f'<td>{caption}</td>',
                        '</tr>',
                    ]
                html += ['</table>']
        html += ['</div>']
    html += ['</body>', '</html>']

    if save_images:
        imgsave.save_image_set(images, str(out_dir / image_file_name_pattern))

    html_file = out_dir / 'index.html'
    with html_file.open('w') as handle:
        handle.writelines(html)
