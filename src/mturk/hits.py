"""Tools for generating MTurk HITS."""
import collections
import csv
import pathlib
import random
from typing import Callable, Mapping, Optional, Sequence
from urllib import request

from src.milannotations import datasets
from src.utils import lang
from src.utils.typing import Layer, PathLike, StrSequence

import spellchecker
from tqdm.auto import tqdm


def generate_hits_csv(
    dataset: datasets.TopImagesDataset,
    csv_file: PathLike,
    generate_urls: Callable[[Layer, int, int], Sequence[str]],
    validate_urls: bool = True,
    limit: Optional[int] = None,
    layer_column: str = 'layer',
    unit_column: str = 'unit',
    image_url_column_prefix: str = 'image_url_',
    display_progress: bool = True,
) -> None:
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
        limit (Optional[int], optional): Maximum number of units to generate
            HITS for. Will be sampled at random if set. Defaults to None.
        layer_column (str, optional): Layer column in generated CSV.
            Defaults to 'layer'.
        unit_column (str, optional): Unit column in generated CSV.
            Defaults to 'unit'.
        image_url_column_prefix (str, optional): Prefix for image URL columns.
            Will be postfixed with index of the image in the top images list.
            Defaults to 'image_url_'.
        display_progress (bool, optional): If True, display progress bar.
            Defaults to True.

    Raises:
        ValueError: If URLs do not exist when validate_urls is True, or if
            generate_urls returns too many URLs.

    """
    csv_file = pathlib.Path(csv_file)
    csv_file.parent.mkdir(exist_ok=True, parents=True)

    header = [layer_column, unit_column]
    header += [
        f'{image_url_column_prefix}{index + 1}' for index in range(dataset.k)
    ]

    samples = dataset.samples
    if limit is not None and len(samples) > limit:
        samples = random.sample(samples, k=limit)
    if display_progress:
        samples = tqdm(samples, desc='process samples')

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

        row = [str(layer), str(unit)]
        row += urls
        if len(row) < dataset.k + 2:
            row += [''] * (dataset.k + 2 - len(row))
        rows.append(row)

    with csv_file.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


ResultsRow = Mapping[str, str]
TransformLayer = Callable[[str, ResultsRow], str]
TransformUnit = Callable[[str, ResultsRow], str]
TransformAnnotation = Callable[[str, ResultsRow], str]


def strip_results_csv(
    results_csv_file: PathLike,
    out_csv_file: Optional[PathLike] = None,
    in_layer_column: str = 'Input.layer',
    in_unit_column: str = 'Input.unit',
    in_annotation_column: str = 'Answer.summary',
    in_rejection_column: str = 'RejectionTime',
    out_layer_column: str = 'layer',
    out_unit_column: str = 'unit',
    out_annotation_column: str = 'summary',
    keep_rejected: bool = False,
    spellcheck: bool = False,
    remove_prefixes: Optional[StrSequence] = None,
    remove_substrings: Optional[StrSequence] = None,
    remove_suffixes: Optional[StrSequence] = None,
    replace_prefixes: Optional[Mapping[str, str]] = None,
    replace_substrings: Optional[Mapping[str, str]] = None,
    replace_suffixes: Optional[Mapping[str, str]] = None,
    replace_exact: Optional[Mapping[str, str]] = None,
    transform_layer: Optional[TransformLayer] = None,
    transform_unit: Optional[TransformUnit] = None,
    transform_annotation: Optional[TransformAnnotation] = None,
) -> None:
    """Strip the results CSV of everything but layer, unit, and annotation.

    Args:
        results_csv_file (PathLike): Results CSV downloaded from MTurk.
        out_csv_file (Optional[PathLike], optional): Where to put stripped CSV.
            Defaults to original CSV.
        in_layer_column (str, optional): Layer column in input CSV.
            Defaults to 'Input.layer'.
        in_unit_column (str, optional): Unit column in input CSV.
            Defaults to 'Input.unit'.
        in_annotation_column (str, optional): Annotation column in input CSV.
            Defaults to 'Answer.summary'.
        in_rejection_column (str, optional): Column in input CSV that indicates
            whether HIT was rejected or not. Defaults to 'RejectionTime'.
        out_layer_column (str, optional): Layer column in output CSV.
            Defaults to 'layer'.
        out_unit_column (str, optional): Unit column in output CSV.
            Defaults to 'unit'.
        out_annotation_column (str, optional): Annotation column in output CSV.
            Defaults to 'summary'.
        keep_rejected (bool, optional): If set, keep rejected HITs. Otherwise
            they will be removed. Defaults to False.
        spellcheck (bool, optional): Spellcheck and correct the annotations.
            Defaults to False.
        remove_prefixes (Optional[StrSequence], optional): Remove these
            prefixes from annotations in this order. Defaults to None.
        remove_substrings (Optional[StrSequence], optional): Remove these
            substrings from annotations in this order. Defaults to None.
        remove_suffixes (Optional[StrSequence], optional): Remove these
            suffixes from annotations in this order. Defaults to None.
        replace_prefixes (Optional[Mapping[str, str]], optional): Replace
            prefixes (keys) with new strings (values). Defaults to None.
        replace_substrings (Optional[Mapping[str, str]], optional): Replace
            substrings (keys) with new strings (values). Defaults to None.
        replace_suffixes (Optional[Mapping[str, str]], optional): Replace
            suffixes (keys) with new strings (values). Defaults to None.
        replace_exact (Optional[Mapping[str, str]], optional): Replace string
            (key) with value if it exactly matches the annotation.
            Defaults to None.
        transform_layer (Optional[TransformLayer], optional): Apply this
            function before processing layer. Function takes both the layer
            and CSV row as arguments. Defaults to None.
        transform_unit (Optional[TransformUnit], optional): Apply this
            function before processing unit. Function takes both the unit
            and CSV row as arguments. Defaults to None.
        transform_annotation (Optional[TransformAnnotation], optional): Apply
            this function to annotation after processing it (e.g., to make
            worker-specific corrections to it). Function takes both the
            annotation and the CSV row as arguments. Defaults to None.

    """
    results_csv_file = pathlib.Path(results_csv_file)
    if not results_csv_file.is_file():
        raise FileNotFoundError(f'file not found: {results_csv_file}')

    if out_csv_file is None:
        out_csv_file = results_csv_file
    out_csv_file = pathlib.Path(out_csv_file)
    out_csv_file.parent.mkdir(exist_ok=True, parents=True)

    with results_csv_file.open('r') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None, 'null columns?'
        fields = set(reader.fieldnames)
        inputs = tuple(reader)

    for column in (in_layer_column, in_unit_column, in_annotation_column,
                   in_rejection_column):
        if column not in fields:
            raise KeyError(f'mturk results csv missing column: {column}')

    # Set up all of our cleaning dictionaries.
    replace_prefixes = collections.OrderedDict(replace_prefixes or {})
    if remove_prefixes is not None:
        for prefix in remove_prefixes:
            replace_prefixes[prefix] = ''

    replace_substrings = collections.OrderedDict(replace_substrings or {})
    if remove_substrings is not None:
        for substring in remove_substrings:
            replace_substrings[substring] = ''

    replace_suffixes = collections.OrderedDict(replace_suffixes or {})
    if remove_suffixes is not None:
        for suffix in remove_suffixes:
            replace_suffixes[suffix] = ''

    replace_exact = collections.OrderedDict(replace_exact or {})

    if spellcheck:
        spell = spellchecker.SpellChecker()
        vocab = lang.vocab([input[in_annotation_column] for input in inputs],
                           tokenize=lang.tokenizer(lemmatize=False,
                                                   ignore_stop=False,
                                                   ignore_punct=False))
        for word in tqdm(spell.unknown(vocab.tokens), desc='spellchecking'):
            correction = spell.correction(word)

            for punct in (' ', ',', '--', '-', "'", '"', ':', ';'):
                key = f'{word}{punct}'
                if key in replace_prefixes:
                    continue
                replace_prefixes[key] = f'{correction}{punct}'

            for punct in (' ', ',', '.', "'", '"', '--', '-'):
                key = f' {word}{punct}'
                if key in replace_substrings:
                    continue
                replace_substrings[key] = f' {correction}{punct}'

            for punct in ('', '.', "'"):
                key = f' {word}{punct}'
                if key in replace_suffixes:
                    continue
                replace_suffixes[key] = f' {correction}{punct}'

            if word not in replace_exact:
                replace_exact[word] = correction

    # Okay, now construct the output CSV.
    header = (out_layer_column, out_unit_column, out_annotation_column)
    outputs = [header]
    for input in inputs:
        if not keep_rejected and input[in_rejection_column].strip():
            continue

        # Read layer.
        layer = input[in_layer_column]
        if transform_layer is not None:
            layer = transform_layer(layer, input)

        # Read unit.
        unit = input[in_unit_column]
        if transform_unit is not None:
            unit = transform_unit(unit, input)

        # Read annotation. We always lowercase it before cleaning.
        annotation = input[in_annotation_column].lower()
        for prefix, replacement in replace_prefixes.items():
            if annotation.startswith(prefix):
                annotation = replacement + annotation[len(prefix):]
        for substring, replacement in replace_substrings.items():
            annotation = annotation.replace(substring, replacement)
        for suffix, replacement in replace_suffixes.items():
            if annotation.endswith(suffix):
                annotation = annotation[:-len(suffix)] + replacement
        for string, replacement in replace_exact.items():
            if annotation == string:
                annotation = replacement
        annotation = annotation.strip()

        # Apply the transformation after doing all these other fixes.
        if transform_annotation is not None:
            annotation = transform_annotation(annotation, input)

        # Add row to output.
        output = (layer, unit, annotation)
        outputs.append(output)

    with out_csv_file.open('w') as handle:
        writer = csv.writer(handle)
        writer.writerows(outputs)
