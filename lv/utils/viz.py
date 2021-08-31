"""Visualization utilities."""
import collections
import pathlib
import random
from typing import Any, Callable, Optional, Sequence, Sized, Tuple, Union, cast

from lv import datasets
from lv.utils.typing import StrSequence, PathLike

import wandb
from PIL import Image
from torch.utils import data


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
            dataset index and sample to a PIL image. By default, tries to
            call `sample.as_pil_image_grid()` to be compatible with
            TopImagesDataset and its variants.
        sample_to_caption (Callable[[int, Any], str]): Function that maps
            dataset index and sample to caption. Defaults to None.
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


AnyTopImages = Union[datasets.TopImages, datasets.AnnotatedTopImages]


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


def wandb_dist_plot(values: Sequence[Any],
                    normalize: bool = True,
                    columns: Optional[Tuple[str, str]] = None,
                    title: Optional[str] = None) -> wandb.viz.CustomChart:
    """Create a dist plot of the given values.

    Args:
        values (Sequence[Any]): Values to compute distribution over. Values are
            all converted to strings.
        counts (bool, optional): Normalize counts into fractions.
            Defaults to True.
        columns (Tuple[str, str], optional): Column names.
            Defaults to ('value', 'fraction') if normalize=True and
            ('value', 'counts') otherwise.
        title (Optional[str], optional): Plot title. Defaults to no title.

    Returns:
        wandb.viz.CustomChart: The wandb plot.

    """
    if columns is None:
        columns = ('value', 'fractions' if normalize else 'counts')

    values = [str(value) for value in values]
    counts = collections.Counter(values).most_common()

    dist = [(val, float(n)) for val, n in counts]
    if normalize:
        dist = [(val, n / len(values)) for val, n in dist]

    table = wandb.Table(data=sorted(dist, key=lambda item: item[0]),
                        columns=list(columns))
    return wandb.plot.bar(table, *columns, title=title)


def generate_html(
    dataset: data.Dataset[AnyTopImages],
    out_dir: PathLike,
    captions: Optional[StrSequence] = None,
    get_header: Optional[Callable[[AnyTopImages], str]] = None,
    get_image_url: Optional[Callable[[AnyTopImages], str]] = None,
    include_gt: bool = True,
) -> None:
    """Generate an HTML visualization of neuron top images and captions.

    Args:
        dataset (data.Dataset[AnyTopImages]): Dataset of neurons.
        out_dir (PathLike): Directory to write top images and final HTML file.
        captions (Optional[StrSequence], optional): Predicted captions to show
            for each neuron. Defaults to None.
        get_header (Optional[Callable[[AnyTopImages], str]], optional): Fn
            returning the header text for each neuron. By default, header is
            "{layer}-{unit}".
        get_image_url (Optional[Callable[[AnyTopImages], str]], optional): Fn
            returning the URL of top images for a given neuron. By default,
            images are saved to out_dir and image URLs are local file system
            paths.
        include_gt (bool, optional): If set, also write ground truth captions
            to the HTML when possible. Defaults to True.

    Raises:
        ValueError: If `captions` is set but has different length
            than `dataset`.

    """
    length = len(cast(Sized, dataset))
    if captions is not None and len(captions) != length:
        raise ValueError(f'expected {length} captions, '
                         f'got {len(captions)}')

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    html = [
        '<!doctype html>',
        '<html>',
        '<body>',
    ]
    for index in range(length):
        sample = dataset[index]
        key = f'{sample.layer}-{sample.unit}'

        if get_header is not None:
            header = get_header(sample)
        else:
            header = key

        if get_image_url is not None:
            image_url = get_image_url(sample)
        else:
            image_url = str(out_dir.absolute() / f'{key}.png')
            sample.as_pil_image_grid().save(image_url)

        html += [
            '<div>',
            f'<h3>{header}</h3>',
            f'<img src="{image_url}/>',
        ]

        if include_gt and isinstance(sample, datasets.AnnotatedTopImages):
            html += ['<h5>human annotations</h5>', '<ul>']
            for annotation in sample.annotations:
                html += [f'<li>{annotation}</li>']
            html += ['</ul>']

        if captions is not None:
            html += [
                '<h5>predicted caption</h5>',
                captions[index],
            ]

        html += ['</div>']

    html += ['</body>', '</html>']

    html_file = out_dir / 'index.html'
    with html_file.open('w') as handle:
        handle.writelines(html)
