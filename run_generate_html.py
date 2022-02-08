"""Generate HTML summary of top-images and captions."""
import argparse
import csv
import pathlib

import src.datasets
from src import models, zoo
from src.utils import env, viz
from src.utils.typing import StrSequence

from torch import cuda

parser = argparse.ArgumentParser(description='generate html page of captions')
parser.add_argument('captioner',
                    help='captioner variant (e.g. captioner-resnet101)')
parser.add_argument('train', help='data captioner was trained on (e.g. all)')
parser.add_argument(
    'test',
    help='dataset or dataset group to caption (e.g. dino_vit8/imagenet)')
parser.add_argument('--results-dir',
                    type=pathlib.Path,
                    help='where to write html (default: project results dir)')
parser.add_argument('--base-url',
                    default='https://unitname.csail.mit.edu/generated-html',
                    help='base url for images (default: csail url)')
parser.add_argument('--grid-images',
                    action='store_true',
                    help='save images as grids')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

base_url = args.base_url.rstrip('/')

# Load model.
decoder, _ = zoo.model(args.captioner, args.train, map_location=device)
assert isinstance(decoder, models.Decoder)

# Load dataset(s).
if args.test in zoo.DATASET_GROUPINGS:
    datasets = {}
    for key in zoo.DATASET_GROUPINGS[args.test]:
        dataset = zoo.dataset(key)
        assert isinstance(dataset, (
            src.datasets.TopImagesDataset,
            src.datasets.AnnotatedTopImagesDataset,
        ))
        datasets[key] = dataset
else:
    dataset = zoo.dataset(args.test)
    assert isinstance(dataset, (
        src.datasets.TopImagesDataset,
        src.datasets.AnnotatedTopImagesDataset,
    ))
    datasets = {args.test: dataset}

# Prepare results dir.
results_dir = args.results_dir or (env.results_dir() / 'generated-html')
results_dir.mkdir(exist_ok=True, parents=True)

# If necessary, save images. To avoid doing this multiple times, save all
# images under special dir indexed by test dataset.
for key, dataset in datasets.items():
    images_subdir = f'images/{key.replace("/", "-")}'
    images_dir = results_dir / images_subdir
    if not images_dir.exists():
        images_dir.mkdir(exist_ok=True, parents=True)
        viz.generate_html(
            dataset,
            images_dir,
            get_base_url=lambda *_: f'{base_url}/{images_subdir}',
            include_gt=True,
            save_images=True,
            grid_images=args.grid_images)

# Prepare HTML output directory.
html_subdir = f'{args.captioner}-{args.train}/{args.test.replace("/", "-")}'
html_dir = results_dir / html_subdir
html_dir.mkdir(exist_ok=True, parents=True)

# Combine all the datasets into one, keeping track of where each one
# came from so we can set the URLs properly.
key, dataset = next(iter(datasets.items()))
keys = [key] * len(dataset)
ids = list(range(len(dataset)))
for other in datasets.keys() - {key}:
    dataset += datasets[other]
    keys += [other] * len(datasets[other])
    ids += range(len(datasets[other]))

# Run the captioner, or check if we already did.
captions_file = html_dir / 'captions.csv'
if captions_file.exists():
    print(f'loading captions from {captions_file}')
    with captions_file.open('r') as handle:
        rows = tuple(csv.DictReader(handle))
    predictions: StrSequence = [row['caption'] for row in rows]
else:
    predictions = decoder.predict(dataset,
                                  strategy='rerank',
                                  temperature=.2,
                                  beam_size=50,
                                  device=device)

    # Save them for later.
    outputs = [('layer', 'unit', 'caption')]
    for index, caption in enumerate(predictions):
        sample = dataset[index]
        outputs.append((sample.layer, str(sample.unit), caption))
    print(f'saving captions to {captions_file}')
    with captions_file.open('w') as handle:
        csv.writer(handle).writerows(outputs)

# Save the HTML results.
viz.generate_html(dataset,
                  html_dir,
                  predictions=predictions,
                  get_base_url=lambda _, index:
                  f'{base_url}/images/{keys[index].replace("/", "-")}',
                  get_unit_id=lambda _, index: ids[index],
                  include_gt=True,
                  save_images=False,
                  grid_images=args.grid_images)
