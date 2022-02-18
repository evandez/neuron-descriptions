"""Download blurred ILSVRC models from Google Drive."""
import argparse
import collections
import pathlib
from typing import Dict

from src import milannotations
from src.utils import env

import gdown
import torch

GDRIVE_BASE_URL = 'https://drive.google.com/uc?id='
GDRIVE_IDS = {
    milannotations.KEYS.ALEXNET:
        '1BmgExiP10P5j0irBiCf2TVwFpNnXTVOv',
    milannotations.KEYS.DENSENET121:
        '1yeKuiREpdl9ltyVQEcfzgAcjkfK9Punj',
    milannotations.KEYS.DENSENET201:
        '1s3lGJ8Lq67LVgpa9nArUnt-Augfnd7mP',
    milannotations.KEYS.MOBILENET_V2:
        '1DJIgaQVsRroY1TInBzqenXwXTQR6X4hk',
    milannotations.KEYS.RESNET18:
        '1woDKMm90armYrOZ9lfXTg-MWhWNA0eDD',
    milannotations.KEYS.RESNET34:
        '10Kqkr3ULhzV_llN6lgBfZ4TOd6uz-iDW',
    milannotations.KEYS.RESNET50:
        '1dmT7HVyTp8OwFEbgIDN6P5RNuvsLpWEC',
    milannotations.KEYS.RESNET101:
        '1tnG1gKRL2VrXMS_zD09KERFt9nbzuXq_',
    milannotations.KEYS.RESNET152:
        '1LxrgwDKijRqBAxy9odPqKRELASBLFBOa',
    milannotations.KEYS.SHUFFLENET_V2_X1_0:
        '1ifWeFumTS9Kjbvq0hm1hpVrJaYgJFg5Y',
    milannotations.KEYS.SQUEEZENET1_0:
        '15Ro0jRzpk9-5q_U-rKZlMpe4jHLg_jXi',
    milannotations.KEYS.VGG11:
        '1AhzaMsxTpM08Q22sp94aF2U-Xerd4uE5',
    milannotations.KEYS.VGG13:
        '1fyNnwpath6_BcfgtqaZ1DRnyZr4QhKFe',
    milannotations.KEYS.VGG16:
        '18hyyLVplUZUi2u1_Y-MYsJ5JL3aG7n3H',
    milannotations.KEYS.VGG19:
        '1FhvooAy-ahtX_vyoxwlmOi8VJlj7JE-j',
}

parser = argparse.ArgumentParser(description='download blurred ilsvrc models')
parser.add_argument('--models-dir',
                    type=pathlib.Path,
                    help='save converted models to this directory')
parser.add_argument('--no-cache',
                    action='store_true',
                    help='force redownload models even if cached')
args = parser.parse_args()

models_dir = args.models_dir or env.models_dir()
models_dir.mkdir(exist_ok=True, parents=True)

downloaded_files: Dict[str, str] = {}
for name, gdrive_id in GDRIVE_IDS.items():
    gdrive_url = GDRIVE_BASE_URL + gdrive_id
    if args.no_cache:
        downloaded_file = gdown.download(gdrive_url)
    else:
        downloaded_file = gdown.cached_download(gdrive_url)
    downloaded_files[name] = downloaded_file

for name, file in downloaded_files.items():
    state_dict = torch.load(file, map_location='cpu')['state_dict']

    # A lot of these models were wrapped with DataParallel, so we have
    # to remap the weights.
    state_dict_remapped = collections.OrderedDict()
    for param_key, param_value in state_dict.items():
        param_key = param_key.replace('module.', '')
        state_dict_remapped[param_key] = param_value

    model_file_name = f'{name}-{milannotations.KEYS.IMAGENET_BLURRED}.pth'
    model_file = models_dir / model_file_name
    print(f'saving {name} blurred imagenet model to {model_file}')
    torch.save(state_dict_remapped, model_file)
