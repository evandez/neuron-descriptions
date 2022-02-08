"""Generate captions for a bunch of models."""
import argparse
import csv
import pathlib
import shutil

import src.datasets
from src import milan, zoo
from src.utils import env

from torch import cuda

CNNS = (
    # zoo.KEYS.ALEXNET_IMAGENET,
    # zoo.KEYS.ALEXNET_IMAGENET_BLURRED,
    zoo.KEYS.DENSENET121_IMAGENET,
    zoo.KEYS.DENSENET121_IMAGENET_BLURRED,
    zoo.KEYS.DENSENET201_IMAGENET,
    zoo.KEYS.DENSENET201_IMAGENET_BLURRED,
    zoo.KEYS.MOBILENET_V2_IMAGENET,
    zoo.KEYS.MOBILENET_V2_IMAGENET_BLURRED,
    zoo.KEYS.RESNET18_IMAGENET,
    zoo.KEYS.RESNET18_IMAGENET_BLURRED,
    zoo.KEYS.RESNET34_IMAGENET,
    zoo.KEYS.RESNET34_IMAGENET_BLURRED,
    zoo.KEYS.RESNET50_IMAGENET,
    zoo.KEYS.RESNET50_IMAGENET_BLURRED,
    # zoo.KEYS.RESNET101_IMAGENET,
    # zoo.KEYS.RESNET101_IMAGENET_BLURRED,
    # zoo.KEYS.RESNET152_IMAGENET,
    # zoo.KEYS.RESNET152_IMAGENET_BLURRED,
    zoo.KEYS.SQUEEZENET1_0_IMAGENET,
    zoo.KEYS.SQUEEZENET1_0_IMAGENET_BLURRED,
    zoo.KEYS.SHUFFLENET_V2_X1_0_IMAGENET,
    zoo.KEYS.SHUFFLENET_V2_X1_0_IMAGENET_BLURRED,
    zoo.KEYS.VGG11_IMAGENET,
    zoo.KEYS.VGG11_IMAGENET_BLURRED,
    zoo.KEYS.VGG13_IMAGENET,
    zoo.KEYS.VGG13_IMAGENET_BLURRED,
    zoo.KEYS.VGG16_IMAGENET,
    zoo.KEYS.VGG16_IMAGENET_BLURRED,
    zoo.KEYS.VGG19_IMAGENET,
    zoo.KEYS.VGG19_IMAGENET_BLURRED,
)

parser = argparse.ArgumentParser(
    description='audit cnns by captioning all neurons')
parser.add_argument('--captioner',
                    nargs=2,
                    default=(zoo.KEYS.CAPTIONER_RESNET101, zoo.KEYS.ALL),
                    help='captioner to use (default: captioner-resnet101 all)')
parser.add_argument('--cnns',
                    nargs='+',
                    choices=CNNS,
                    default=CNNS,
                    help='models to audit (default: all)')
parser.add_argument('--data-dir',
                    type=pathlib.Path,
                    help='root dir containing models to audit '
                    '(default: <project data dir> / <cnn key>)')
parser.add_argument('--results-dir',
                    type=pathlib.Path,
                    help='root dir for intermediate and final results '
                    '(default: project results dir)')
parser.add_argument('--clear-results-dir',
                    action='store_true',
                    help='if set, clear results dir (default: do not)')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

decoder, _ = zoo.model(*args.captioner)
decoder.to(device)
assert isinstance(decoder, milan.Decoder)

results_dir = args.results_dir or (env.results_dir() / 'cnn-audit')
results_dir.mkdir(exist_ok=True, parents=True)
if args.clear_results_dir and results_dir.exists():
    shutil.rmtree(results_dir)
results_dir.mkdir(exist_ok=True, parents=True)

for key in args.cnns:
    print(f'---- audit {key} ----')

    captions_file = results_dir / f'{key.replace("/", "-")}-captions.csv'
    if captions_file.exists():
        print(f'found captions file at {captions_file}; skipping')
        continue

    path = None
    if args.data_dir is not None:
        path = args.data_dir / key

    dataset = zoo.dataset(key, path=path)
    assert isinstance(dataset, src.datasets.TopImagesDataset)

    predictions = decoder.predict(dataset,
                                  strategy='rerank',
                                  temperature=.2,
                                  beam_size=50,
                                  device=device)

    rows = [('layer', 'unit', 'caption')]
    for index, caption in enumerate(predictions):
        sample = dataset[index]
        rows.append((sample.layer, str(sample.unit), caption))
    print(f'saving captions to {captions_file}')
    with captions_file.open('w') as handle:
        csv.writer(handle).writerows(rows)
