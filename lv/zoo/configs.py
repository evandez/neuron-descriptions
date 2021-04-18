"""Zoo configs for latent vocabulary models.

This module contains configs for all models trained on latent vocabulary data.
If you're looking for the original models--whose neurons we've annotated as
part of this project--you are at the wrong zoo. You want to look at the configs
in lv/dissection/zoo.py instead.
"""
from lv.zoo import core


def models() -> core.ModelConfigs:
    """Return all model configs."""
    return {}


def datasets() -> core.DatasetConfigs:
    """Return all dataset configs."""
    return {}
