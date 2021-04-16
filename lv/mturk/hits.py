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
                      generate_urls: Callable[[str, int, int], Sequence[str]],
                      validate_urls: bool = True,
                      display_progress: bool = True) -> None:
    """Generate MTurk hits CSV file for the given dataset.

    Each (layer, unit) gets its own hit. The CSV will have the format:

        layer,unit,image_url_1,...,image_url_k
        "my-layer-1","my-unit-1","https://images.com/unit-1-image-1.png",...

    The caller must specify how to create the URLs for each layer and unit,
    as this library does not provide any tools for hosting images.

    Args:
        dataset (datasets.TopImagesDataset): Dataset to generate hits for.
        csv_file (PathLike): File to write hits to.
        generate_urls (Callable[[str, int], Sequence[str]]): Function taking
            layer, unit, and number of top images as input and returning
            all URLs.
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

    header = ['layer', 'unit']
    header += [f'image_url_{index + 1}' for index in range(dataset.k)]

    samples = dataset.samples
    if display_progress:
        samples = tqdm.tqdm(samples, desc=f'processing {len(samples)} samples')

    rows = [header]
    for layer, unit, *_ in samples:
        urls = generate_urls(layer, unit, dataset.k)
        if len(urls) > dataset.k:
            raise ValueError(f'generate_urls returned {len(urls)} '
                             f'but each unit has <= {dataset.k}')

        if validate_urls:
            for url in urls:
                code = request.urlopen(url).getcode()
                if code != 200:
                    raise ValueError(f'bad url (code {code}): {url}')

        row = [layer, str(unit)]
        row += urls
        if len(row) < dataset.k + 2:
            row += [''] * (dataset.k + 2 - len(row))
        rows.append(row)

    with csv_file.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
