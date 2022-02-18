"""Tools for loading pretrained MILAN models."""
from typing import Any

import src.milannotations.loaders
from src.milan import decoders
from src.utils import hubs


def hub() -> hubs.ModelHub:
    """Create the model hub."""
    return hubs.ModelHub(
        **{
            group: hubs.ModelConfig(
                decoders.Decoder.load,
                url=f'{hubs.HOST}/models/{group.replace("/", "_")}.pth',
                requires_path=True,
                load_weights=False,
            ) for group in src.milannotations.loaders.DATASET_GROUPINGS
        })


def pretrained(config: str = 'base', **kwargs: Any) -> decoders.Decoder:
    """Return a pretrained MILAN model."""
    model = hub().load(config, **kwargs)
    assert isinstance(model, decoders.Decoder), model
    return model
