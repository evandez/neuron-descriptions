"""Tools for generating MTurk HITS."""
import csv
import pathlib
from typing import Callable, Sequence
from urllib import request

from lv import datasets
from lv.typing import PathLike

import tqdm


def generate_hits_csv(dataset: datasets.TopImagesDataset,
                      csv_file: PathLike,
                      generate_urls: Callable[[str, str], Sequence[str]],
                      validate_urls: bool = True,
                      display_progress: bool = True) -> None:
    """Generate MTurk hits CSV file for the given dataset.

    Each (layer, unit) gets its own hit. The CSV will have the format:

        layer,unit,image_url_1,...,image_url_k
        "my-layer-1","my-unit-1","https://images.com/unit-1-image-1.png",...

    If some units have fewer top images than others, the CSV will pad the row
    with empty strings. While layer/unit is not displayed to MTurk workers, it
    is carried over to the results CSV as metadata and is useful to include.

    The caller must specify how to create the URLs for each layer and unit,
    as this library does not provide any tools for hosting images.

    Args:
        dataset (datasets.TopImagesDataset): Dataset to generate hits for.
        csv_file (PathLike): File to write hits to.
        generate_urls (Callable[[str, str], Sequence[str]]): Function taking
            layer and unit as input and returning all URLs
        validate_urls (bool, optional): If set, make sure all image URLs
            actually open. Defaults to True.
        display_progress (bool, optional): If True, display progress bar.
            Defaults to True.

    Raises:
        ValueError: If URLs do not exist when validate_urls is True, or if
            generate_urls returns too many URLs.

    """
    csv_file = pathlib.Path(csv_file)
    csv_file.parent.mkdir(exist_ok=True, parents=True)

    # TopImagesDataset does not require that each unit has the same
    # number of images associated with it, but we need to know
    # how many image_url columns there should be in the CSV. Find the
    # largest number of images any unit has (without reading every image).
    n_images = max([len(indices) for _, _, indices in dataset.samples])

    header = ['layer', 'unit']
    header += [f'image_url_{index + 1}' for index in range(n_images)]

    samples = dataset.samples
    if display_progress:
        samples = tqdm.tqdm(samples, desc=f'processing {len(samples)} samples')

    rows = [header]
    for layer, unit, _ in samples:
        urls = generate_urls(layer, unit)
        if len(urls) > n_images:
            raise ValueError(f'generate_urls returned {len(urls)} '
                             f'but each unit has <= {n_images}')

        if validate_urls:
            for url in urls:
                code = request.urlopen(url).getcode()
                if code != 200:
                    raise ValueError(f'bad url (code {code}): {url}')

        row = [layer, unit]
        row += urls
        if len(row) < n_images + 2:
            row += [''] * (n_images + 2 - len(row))
        rows.append(row)

    with csv_file.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
