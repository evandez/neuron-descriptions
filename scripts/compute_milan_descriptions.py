"""Compute MILAN descriptions for given model/dataset pair.

See compute_exemplars.py for how args should be specified.
"""
import argparse
import csv
import pathlib

from src import milan, milannotations
from src.utils import env

from torch import cuda

parser = argparse.ArgumentParser(description='compute milan descriptions')
parser.add_argument('model', help='model architecture (e.g. alexnet)')
parser.add_argument('dataset', help='dataset model trained on (e.g. imagenet)')
parser.add_argument('--temperature',
                    type=float,
                    default=.2,
                    help='pmi temperature (default: .2)')
parser.add_argument('--beam-size',
                    type=int,
                    default=50,
                    help='beam size to rerank (default: 50)')
parser.add_argument('--data-dir',
                    type=pathlib.Path,
                    help='root dir for datasets (default: project data dir)')
parser.add_argument(
    '--results-dir',
    type=pathlib.Path,
    help='root dir for final results '
    '(default: <project results dir> / descriptions / model_dataset.csv)')
parser.add_argument('--milan',
                    default=milannotations.KEYS.BASE,
                    help='milan model to use (default: base)')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

# Prepare directories.
key = f'{args.model}/{args.dataset}'
data_dir = args.data_dir or env.data_dir()
data_root = data_dir / key

results_dir = args.results_dir
if results_dir is None:
    results_dir = env.results_dir() / 'descriptions'
results_dir.mkdir(exist_ok=True, parents=True)

# Load MILAN
decoder = milan.pretrained(args.milan)
decoder.to(device)

# Load top images.
dataset = milannotations.load(key, path=data_root)

# Go!
predictions = decoder.predict(dataset,
                              strategy='rerank',
                              temperature=args.temperature,
                              beam_size=args.beam_size,
                              device=device)

rows = [('layer', 'unit', 'description')]
for index, description in enumerate(predictions):
    sample = dataset[index]
    row = (str(sample.layer), str(sample.unit), description)
    rows.append(row)
results_csv_file = results_dir / f'{key.replace("/", "_")}.csv'
with results_csv_file.open('w') as handle:
    csv.writer(handle).writerows(rows)
