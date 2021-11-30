"""Export top images and descriptions to a zip file with json metadata."""
import argparse
import json
import pathlib
import shutil
from typing import Dict

from lv import datasets, models, zoo
from lv.utils import env

from torch import cuda
from tqdm.auto import tqdm

SOURCES = (
    zoo.KEYS.DENSENET121_IMAGENET,
    zoo.KEYS.DENSENET121_IMAGENET_BLURRED,
    zoo.KEYS.DENSENET201_IMAGENET,
    zoo.KEYS.DENSENET201_IMAGENET_BLURRED,
    zoo.KEYS.MOBILENET_V2_IMAGENET,
    zoo.KEYS.MOBILENET_V2_IMAGENET_BLURRED,
    zoo.KEYS.SHUFFLENET_V2_X1_0_IMAGENET,
    zoo.KEYS.SHUFFLENET_V2_X1_0_IMAGENET_BLURRED,
    zoo.KEYS.SQUEEZENET1_0_IMAGENET,
    zoo.KEYS.SQUEEZENET1_0_IMAGENET_BLURRED,
    zoo.KEYS.VGG11_IMAGENET,
    zoo.KEYS.VGG11_IMAGENET_BLURRED,
    zoo.KEYS.VGG13_IMAGENET,
    zoo.KEYS.VGG13_IMAGENET_BLURRED,
    zoo.KEYS.VGG16_IMAGENET,
    zoo.KEYS.VGG16_IMAGENET_BLURRED,
    zoo.KEYS.VGG19_IMAGENET,
    zoo.KEYS.VGG19_IMAGENET_BLURRED,
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
                    help='models to caption and export (default: all)')
parser.add_argument('--captioner',
                    nargs=2,
                    default=(zoo.KEYS.CAPTIONER_RESNET101, zoo.KEYS.ALL),
                    help='captioner to use (default: captioner-resnet101 all)')
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

decoder, _ = zoo.model(*args.captioner)
decoder.to(device)
assert isinstance(decoder, models.Decoder)

# Read in all datasets.
data: Dict[str, datasets.TopImagesDataset] = {}
for key in args.sources:
    dataset = zoo.dataset(key,
                          factory=datasets.TopImagesDataset,
                          path=data_dir / key)
    assert isinstance(dataset, datasets.TopImagesDataset)
    data[key] = dataset

# Caption all the data.
captions = {}
for key in args.sources:
    captions_file = results_dir / f'cache/{key.replace("/", "_")}_captions.csv'
    if captions_file.exists():
        print(f'reading {key} captions from {captions_file}')
        with captions_file.open('r') as handle:
            captions[key] = tuple(handle.read().split('\n'))
    else:
        predictions = decoder.predict(data[key],
                                      strategy='rerank',
                                      temperature=.2,
                                      beam_size=50,
                                      device=device)
        print(f'writing {key} captions to {captions_file}')
        captions_file.parent.mkdir(exist_ok=True, parents=True)
        with captions_file.open('w') as handle:
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
                f'{args.base_url}/{image_file.name}'
                for image_file in image_files
            ],
            'description': captions[key][index],
        })

    model_json_file = json_dir / name / 'data.json'
    model_json_file.parent.mkdir(exist_ok=True, parents=True)
    with model_json_file.open('w') as handle:
        json.dump(exported, handle)
