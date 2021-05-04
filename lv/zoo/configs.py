"""Zoo configs for latent vocabulary models.

This module contains configs for all models trained on latent vocabulary data.
If you're looking for the original models--whose neurons we've annotated as
part of this project--you are at the wrong zoo. You want to look at the configs
in lv/dissection/zoo.py instead.
"""
import lv.datasets
from lv.zoo import core

HOST = 'http://wednesday.csail.mit.edu/dez/latent-vocabulary'


def models() -> core.ModelConfigs:
    """Return all model configs."""
    return {}


def datasets() -> core.DatasetConfigs:
    """Return all dataset configs."""
    return {
        'alexnet-imagenet':
            core.DatasetConfig(lv.datasets.TopImagesDataset,
                               url=f'{HOST}/datasets/alexnet-imagenet.zip'),
        'alexnet-imagenet-annotations':
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/datasets/alexnet-imagenet.zip'),
        'alexnet-places365':
            core.DatasetConfig(lv.datasets.TopImagesDataset,
                               url=f'{HOST}/datasets/alexnet-places365.zip'),
        'alexnet-places365-annotations':
            core.DatasetConfig(lv.datasets.AnnotatedTopImagesDataset,
                               url=f'{HOST}/datasets/alexnet-places365.zip'),
    }
