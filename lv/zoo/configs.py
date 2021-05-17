"""Zoo configs for latent vocabulary models.

This module contains configs for all models trained on latent vocabulary data.
If you're looking for the original models--whose neurons we've annotated as
part of this project--you are at the wrong zoo. You want to look at the configs
in lv/dissection/zoo.py instead.
"""
import lv.datasets
from lv.zoo import core

HOST = 'https://unitname.csail.mit.edu'

KEY_ALEXNET = 'alexnet'
KEY_RESNET152 = 'resnet152'
KEY_BIGGAN = 'biggan'

KEY_IMAGENET = 'imagenet'
KEY_PLACES365 = 'places365'

KEY_ALEXNET_IMAGENET = f'{KEY_ALEXNET}/{KEY_IMAGENET}'
KEY_ALEXNET_PLACES365 = f'{KEY_ALEXNET}/{KEY_PLACES365}'
KEY_RESNET152_IMAGENET = f'{KEY_RESNET152}/{KEY_IMAGENET}'
KEY_RESNET152_PLACES365 = f'{KEY_RESNET152}/{KEY_PLACES365}'
KEY_BIGGAN_IMAGENET = f'{KEY_BIGGAN}/{KEY_IMAGENET}'
KEY_BIGGAN_PLACES365 = f'{KEY_BIGGAN}/{KEY_PLACES365}'


def models() -> core.ModelConfigs:
    """Return all model configs."""
    return {}


def datasets() -> core.DatasetConfigs:
    """Return all dataset configs."""
    return {
        KEY_ALEXNET_IMAGENET:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/alexnet-imagenet.zip'),
        KEY_ALEXNET_PLACES365:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/alexnet-places365.zip'),
        KEY_RESNET152_IMAGENET:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/resnet152-imagenet.zip'),
        KEY_RESNET152_PLACES365:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/resnet152-places365.zip'),
        KEY_BIGGAN_IMAGENET:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/biggan-imagenet.zip'),
        KEY_BIGGAN_PLACES365:
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/biggan-places365.zip'),
    }
