"""Export top images and descriptions to a zip file with json metadata."""
import argparse
import json
import pathlib
import shutil
from typing import Dict

from src import milan, milannotations
from src.milannotations import datasets
from src.utils import env

from torch import cuda
from tqdm.auto import tqdm

SOURCES = (
    milannotations.KEYS.DENSENET121_IMAGENET,
    milannotations.KEYS.DENSENET121_IMAGENET_BLURRED,
    milannotations.KEYS.DENSENET201_IMAGENET,
    milannotations.KEYS.DENSENET201_IMAGENET_BLURRED,
    milannotations.KEYS.MOBILENET_V2_IMAGENET,
    milannotations.KEYS.MOBILENET_V2_IMAGENET_BLURRED,
    milannotations.KEYS.SHUFFLENET_V2_X1_0_IMAGENET,
    milannotations.KEYS.SHUFFLENET_V2_X1_0_IMAGENET_BLURRED,
    milannotations.KEYS.SQUEEZENET1_0_IMAGENET,
    milannotations.KEYS.SQUEEZENET1_0_IMAGENET_BLURRED,
    milannotations.KEYS.VGG11_IMAGENET,
    milannotations.KEYS.VGG11_IMAGENET_BLURRED,
    milannotations.KEYS.VGG13_IMAGENET,
    milannotations.KEYS.VGG13_IMAGENET_BLURRED,
    milannotations.KEYS.VGG16_IMAGENET,
    milannotations.KEYS.VGG16_IMAGENET_BLURRED,
    milannotations.KEYS.VGG19_IMAGENET,
    milannotations.KEYS.VGG19_IMAGENET_BLURRED,
)

parser = argparse.ArgumentParser(description='export descriptions')
parser.add_argument('--data-dir',
                    type=pathlib.Path,
                    help='root dir for datasets (default: project data dir)')
parser.add_argument('--results-dir',
                    type=pathlib.Path,
                    help='root dir for intermediate and final results '
                    '(default: project results dir)')
parser.add_argument('--clear-results-dir',
                    action='store_true',
                    help='if set, clear results dir (default: do not)')
parser.add_argument('--sources',
                    nargs='+',
                    default=SOURCES,
                    help='models to describe and export (default: all)')
parser.add_argument('--milan',
                    default=milannotations.KEYS.BASE,
                    help='milan model to use (default: base)')
parser.add_argument('--base-url',
                    default='https://unitname.csail.mit.edu/catalog',
                    help='base url for images (default: csail url)')
parser.add_argument('--no-save-images',
                    action='store_true',
                    help='do not save top images')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

# Prepare directories.
data_dir = args.data_dir or env.data_dir()

results_dir = args.results_dir
if results_dir is None:
    results_dir = env.results_dir() / 'export'

if args.clear_results_dir and results_dir.exists():
    shutil.rmtree(results_dir)
results_dir.mkdir(exist_ok=True, parents=True)

decoder = milan.pretrained(args.milan)
decoder.to(device)

# Read in all datasets.
data: Dict[str, datasets.TopImagesDataset] = {}
for key in args.sources:
    dataset = milannotations.load(key,
                                  factory=datasets.TopImagesDataset,
                                  path=data_dir / key)
    assert isinstance(dataset, datasets.TopImagesDataset)
    data[key] = dataset

# Caption all the data.
descriptions = {}
for key in args.sources:
    descriptions_file = (results_dir /
                         f'cache/{key.replace("/", "_")}_descriptions.csv')
    if descriptions_file.exists():
        print(f'reading {key} descriptions from {descriptions_file}')
        with descriptions_file.open('r') as handle:
            descriptions[key] = tuple(handle.read().split('\n'))
    else:
        predictions = decoder.predict(data[key],
                                      strategy='rerank',
                                      temperature=.2,
                                      beam_size=50,
                                      device=device)
        print(f'writing {key} descriptions to {descriptions_file}')
        descriptions_file.parent.mkdir(exist_ok=True, parents=True)
        with descriptions_file.open('w') as handle:
            handle.write('\n'.join(predictions))

# Save images and JSON files.
json_dir = results_dir / 'json'
images_dir = results_dir / 'images'
for key, dataset in data.items():
    # TODO(evandez): Make this less hacky.
    name = key.replace('/', '_')
    arch = key.split('/')[0]
    exported = {
        'name': name,
        'architecture': arch,
        'dataset': key[len(arch) + 1:],
        'layers': dataset.layers,
    }
    exported['units'] = units = []

    model_images_dir = images_dir / name
    if not args.no_save_images:
        model_images_dir.mkdir(exist_ok=True, parents=True)
    for index in tqdm(range(len(dataset)), desc=f'save {key} images'):
        sample = dataset[index]
        layer, unit = sample.layer, sample.unit

        images = sample.as_pil_images()
        image_files = []
        for position, image in enumerate(images):
            image_file = model_images_dir / f'{layer}_{unit}_{position}.png'
            if not args.no_save_images:
                image.save(image_file)
            image_files.append(image_file)
        units.append({
            'layer': layer,
            'unit': unit,
            'images': [
                f'{args.base_url}/{name}/{image_file.name}'
                for image_file in image_files
            ],
            'description': descriptions[key][index],
        })

    model_json_file = json_dir / name / 'data.json'
    model_json_file.parent.mkdir(exist_ok=True, parents=True)
    with model_json_file.open('w') as handle:
        json.dump(exported, handle)
