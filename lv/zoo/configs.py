"""Zoo configs for latent vocabulary models.

This module contains configs for all models trained on latent vocabulary data.
If you're looking for the original models--whose neurons we've annotated as
part of this project--you are at the wrong zoo. You want to look at the configs
in lv/dissection/zoo.py instead.
"""
import lv.datasets
import lv.models
from lv.zoo import core

HOST = 'https://unitname.csail.mit.edu'

KEY_ALEXNET = 'alexnet'
KEY_RESNET152 = 'resnet152'
KEY_BIGGAN = 'biggan'
KEY_DINO_VITS8 = 'dino_vits8'

KEY_IMAGENET = 'imagenet'
KEY_PLACES365 = 'places365'

KEY_ALEXNET_IMAGENET = f'{KEY_ALEXNET}/{KEY_IMAGENET}'
KEY_ALEXNET_PLACES365 = f'{KEY_ALEXNET}/{KEY_PLACES365}'
KEY_RESNET152_IMAGENET = f'{KEY_RESNET152}/{KEY_IMAGENET}'
KEY_RESNET152_PLACES365 = f'{KEY_RESNET152}/{KEY_PLACES365}'
KEY_BIGGAN_IMAGENET = f'{KEY_BIGGAN}/{KEY_IMAGENET}'
KEY_BIGGAN_PLACES365 = f'{KEY_BIGGAN}/{KEY_PLACES365}'
KEY_DINO_VITS8_IMAGENET = f'{KEY_DINO_VITS8}/{KEY_IMAGENET}'

KEY_GENERATORS = 'gen'
KEY_CLASSIFIERS = 'cls'
KEY_ALL = 'all'
KEY_NOT_ALEXNET_IMAGENET = f'not-{KEY_ALEXNET}-{KEY_IMAGENET}'
KEY_NOT_ALEXNET_PLACES365 = f'not-{KEY_ALEXNET}-{KEY_PLACES365}'
KEY_NOT_RESNET152_IMAGENET = f'not-{KEY_RESNET152}-{KEY_IMAGENET}'
KEY_NOT_RESNET152_PLACES365 = f'not-{KEY_RESNET152}-{KEY_PLACES365}'
KEY_NOT_BIGGAN_IMAGENET = f'not-{KEY_BIGGAN}-{KEY_IMAGENET}'
KEY_NOT_BIGGAN_PLACES365 = f'not-{KEY_BIGGAN}-{KEY_PLACES365}'

# We can group the datasets of neuron annotations in a bunch of interesting
# ways. Here are the most common, used throughout the project. To load a
# grouping, simply do e.g.:
# >>> import lv.zoo
# >>> group = lv.zoo.DATASET_GROUPS['gen']
# >>> dataset = lv.zoo.datasets(*group)
DATASET_GROUPINGS = {
    KEY_ALL: (
        KEY_ALEXNET_IMAGENET,
        KEY_ALEXNET_PLACES365,
        KEY_RESNET152_IMAGENET,
        KEY_RESNET152_PLACES365,
        KEY_BIGGAN_IMAGENET,
        KEY_BIGGAN_PLACES365,
    ),
    KEY_CLASSIFIERS: (
        KEY_ALEXNET_IMAGENET,
        KEY_ALEXNET_PLACES365,
        KEY_RESNET152_IMAGENET,
        KEY_RESNET152_PLACES365,
    ),
    KEY_GENERATORS: (
        KEY_BIGGAN_IMAGENET,
        KEY_BIGGAN_PLACES365,
    ),
    KEY_IMAGENET: (
        KEY_ALEXNET_IMAGENET,
        KEY_RESNET152_IMAGENET,
        KEY_BIGGAN_IMAGENET,
    ),
    KEY_PLACES365: (
        KEY_ALEXNET_PLACES365,
        KEY_RESNET152_PLACES365,
        KEY_BIGGAN_PLACES365,
    ),
    KEY_ALEXNET: (
        KEY_ALEXNET_IMAGENET,
        KEY_ALEXNET_PLACES365,
    ),
    KEY_RESNET152: (
        KEY_RESNET152_IMAGENET,
        KEY_RESNET152_PLACES365,
    ),
    KEY_BIGGAN: (
        KEY_BIGGAN_IMAGENET,
        KEY_BIGGAN_PLACES365,
    ),
    KEY_NOT_ALEXNET_IMAGENET: (
        KEY_ALEXNET_PLACES365,
        KEY_RESNET152_IMAGENET,
        KEY_RESNET152_PLACES365,
        KEY_BIGGAN_IMAGENET,
        KEY_BIGGAN_PLACES365,
    ),
    KEY_NOT_ALEXNET_PLACES365: (
        KEY_ALEXNET_IMAGENET,
        KEY_RESNET152_IMAGENET,
        KEY_RESNET152_PLACES365,
        KEY_BIGGAN_IMAGENET,
        KEY_BIGGAN_PLACES365,
    ),
    KEY_NOT_RESNET152_IMAGENET: (
        KEY_ALEXNET_IMAGENET,
        KEY_ALEXNET_PLACES365,
        KEY_RESNET152_PLACES365,
        KEY_BIGGAN_IMAGENET,
        KEY_BIGGAN_PLACES365,
    ),
    KEY_NOT_RESNET152_PLACES365: (
        KEY_ALEXNET_IMAGENET,
        KEY_ALEXNET_PLACES365,
        KEY_RESNET152_IMAGENET,
        KEY_BIGGAN_IMAGENET,
        KEY_BIGGAN_PLACES365,
    ),
    KEY_NOT_BIGGAN_IMAGENET: (
        KEY_ALEXNET_IMAGENET,
        KEY_ALEXNET_PLACES365,
        KEY_RESNET152_IMAGENET,
        KEY_RESNET152_PLACES365,
        KEY_BIGGAN_PLACES365,
    ),
    KEY_NOT_BIGGAN_PLACES365: (
        KEY_ALEXNET_IMAGENET,
        KEY_ALEXNET_PLACES365,
        KEY_RESNET152_IMAGENET,
        KEY_RESNET152_PLACES365,
        KEY_BIGGAN_IMAGENET,
    ),
}

KEY_CAPTIONER_RESNET101 = 'captioner-resnet101'


def models() -> core.ModelConfigs:
    """Return all model configs."""
    return {
        KEY_CAPTIONER_RESNET101: {
            dataset: core.ModelConfig(
                lv.models.Decoder.load,
                url=f'{HOST}/models/captioner-resnet101-'
                f'{dataset.replace("/", "_")}.pth',
                requires_path=True,
                load_weights=False,
            ) for dataset in DATASET_GROUPINGS.keys()
        },
    }


def datasets() -> core.DatasetConfigs:
    """Return all dataset configs."""
    return {
        KEY_ALEXNET_IMAGENET:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/alexnet-imagenet.zip',
                               annotation_count=3),
        KEY_ALEXNET_PLACES365:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/alexnet-places365.zip',
                               annotation_count=3),
        KEY_RESNET152_IMAGENET:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/resnet152-imagenet.zip',
                               annotation_count=3),
        KEY_RESNET152_PLACES365:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/resnet152-places365.zip',
                               annotation_count=3),
        KEY_BIGGAN_IMAGENET:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/biggan-imagenet.zip',
                               annotation_count=3),
        KEY_BIGGAN_PLACES365:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/biggan-places365.zip',
                               annotation_count=3),
        KEY_DINO_VITS8_IMAGENET:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/dino_vits8-imagenet.zip',
                               annotation_count=3),
    }
