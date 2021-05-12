"""Zoo configs for latent vocabulary models.

This module contains configs for all models trained on latent vocabulary data.
If you're looking for the original models--whose neurons we've annotated as
part of this project--you are at the wrong zoo. You want to look at the configs
in lv/dissection/zoo.py instead.
"""
import lv.datasets
from lv.zoo import core

HOST = 'https://unitname.csail.mit.edu'


def models() -> core.ModelConfigs:
    """Return all model configs."""
    return {}


def datasets() -> core.DatasetConfigs:
    """Return all dataset configs."""
    return {
        'alexnet-imagenet':
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/alexnet-imagenet.zip'),
        'alexnet-places365':
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/alexnet-places365.zip'),
        'resnet152-imagenet':
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/resnet152-imagenet.zip'),
        'resnet152-places365':
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/resnet152-places365.zip'),
        'biggan-imagenet':
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/biggan-imagenet.zip'),
        'biggan-places365':
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/data/biggan-places365.zip'),
    }
