"""Generate HTML summary of top-images and captions."""
import argparse
import pathlib

import lv.datasets
from lv import models, zoo
from lv.utils import env, viz

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
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

# Load model.
decoder, _ = zoo.model(args.captioner, args.train)
assert isinstance(decoder, models.Decoder)

# Load dataset(s).
if args.test in zoo.DATASET_GROUPINGS:
    datasets = {}
    for key in zoo.DATASET_GROUPINGS[args.test]:
        dataset = zoo.dataset(key)
        assert isinstance(dataset, (
            lv.datasets.TopImagesDataset,
            lv.datasets.AnnotatedTopImagesDataset,
        ))
        datasets[key] = dataset
else:
    dataset = zoo.dataset(args.test)
    assert isinstance(dataset, (
        lv.datasets.TopImagesDataset,
        lv.datasets.AnnotatedTopImagesDataset,
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
        viz.generate_html(dataset,
                          images_dir,
                          get_image_urls=lambda sample, _: [
                              f'{args.base_url.rstrip("/")}/{images_subdir}/'
                              f'{sample.layer}_{sample.unit}.png'
                          ],
                          include_gt=True,
                          save_images=True)

# Predict the captions.
key, dataset = next(iter(datasets.items()))
keys = [key] * len(dataset)
for other in datasets.keys() - {key}:
    dataset += datasets[other]
    keys += [other] * len(datasets[other])
predictions = decoder.predict(dataset,
                              strategy='rerank',
                              temperature=.2,
                              beam_size=50,
                              device=device)

# Save the HTML results.
html_subdir = f'{args.captioner}-{args.train}' / args.test.replace('/', '-')
html_dir = results_dir / html_subdir
html_dir.mkdir(exist_ok=True)
viz.generate_html(dataset,
                  html_dir,
                  predictions=predictions,
                  get_image_urls=lambda sample, index: [
                      f'{args.base_url.rstrip("/")}/images/'
                      f'{keys[index].replace("/", "-")}/'
                      f'{sample.layer}_{sample.unit}.png'
                  ],
                  save_images=False,
                  include_gt=True)
