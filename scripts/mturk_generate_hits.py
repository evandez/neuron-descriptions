"""Generate MTurk hits for top images."""
import argparse
import pathlib
from typing import Sequence

from src import milannotations
from src.mturk import hits
from src.utils.typing import Layer

parser = argparse.ArgumentParser(description='generate mturk hits')
parser.add_argument('dataset', help='name of top images dataset')
parser.add_argument('hits_csv_file', type=pathlib.Path, help='output csv file')
parser.add_argument(
    '--dataset-path',
    type=pathlib.Path,
    help='directory containing dataset (default: .zoo/datasets/<dataset>)')
parser.add_argument(
    '--limit',
    type=int,
    help='only generate hits for this many units (default: None)')
parser.add_argument('--host-url',
                    default='https://unitname.csail.mit.edu/dissect',
                    help='host url for top images (default: csail url)')
parser.add_argument('--no-validate-urls',
                    action='store_true',
                    help='do not validate urls')
parser.add_argument('--no-display-progress',
                    action='store_true',
                    help='do not show progress bar')
args = parser.parse_args()

dataset = milannotations.load(args.dataset,
                              path=args.dataset_path,
                              display_progress=not args.no_display_progress)
if not isinstance(dataset, milannotations.TopImagesDataset):
    raise ValueError(f'bad dataset type: {type(dataset).__name__}')

base_url = f'{args.host_url.strip("/")}/{args.dataset}'


def generate_urls(layer: Layer, unit: int, k: int) -> Sequence[str]:
    """Generate top image URLs."""
    return [
        f'{base_url}/{layer}/unit_{unit}/image_{index}.png'
        for index in range(k)
    ]


args.hits_csv_file.parent.mkdir(parents=True, exist_ok=True)
hits.generate_hits_csv(dataset,
                       args.hits_csv_file,
                       generate_urls,
                       limit=args.limit,
                       validate_urls=not args.no_validate_urls,
                       display_progress=not args.no_display_progress)
