"""Defines the hub for MILANNOTATIONS data."""
from typing import Any

from src.milannotations import datasets
from src.utils import hubs

import easydict
import torch.utils.data

KEYS = easydict.EasyDict()
KEYS.ALEXNET = 'alexnet'
KEYS.BIGGAN = 'biggan'
KEYS.DENSENET121 = 'densenet121'
KEYS.DENSENET201 = 'densenet201'
KEYS.DINO_VITS8 = 'dino_vits8'
KEYS.RESNET18 = 'resnet18'
KEYS.RESNET34 = 'resnet34'
KEYS.RESNET50 = 'resnet50'
KEYS.RESNET101 = 'resnet101'
KEYS.RESNET152 = 'resnet152'
KEYS.VGG11 = 'vgg11'
KEYS.VGG13 = 'vgg13'
KEYS.VGG16 = 'vgg16'
KEYS.VGG19 = 'vgg19'
KEYS.MOBILENET_V2 = 'mobilenet_v2'
KEYS.SHUFFLENET_V2_X1_0 = 'shufflenet_v2_x1_0'
KEYS.SQUEEZENET1_0 = 'squeezenet1_0'

KEYS.IMAGENET = 'imagenet'
KEYS.IMAGENET_BLURRED = 'imagenet-blurred'
KEYS.PLACES365 = 'places365'

KEYS.ALEXNET_IMAGENET = f'{KEYS.ALEXNET}/{KEYS.IMAGENET}'
KEYS.BIGGAN_IMAGENET = f'{KEYS.BIGGAN}/{KEYS.IMAGENET}'
KEYS.DENSENET121_IMAGENET = f'{KEYS.DENSENET121}/{KEYS.IMAGENET}'
KEYS.DENSENET201_IMAGENET = f'{KEYS.DENSENET201}/{KEYS.IMAGENET}'
KEYS.DINO_VITS8_IMAGENET = f'{KEYS.DINO_VITS8}/{KEYS.IMAGENET}'
KEYS.MOBILENET_V2_IMAGENET = f'{KEYS.MOBILENET_V2}/{KEYS.IMAGENET}'
KEYS.RESNET18_IMAGENET = f'{KEYS.RESNET18}/{KEYS.IMAGENET}'
KEYS.RESNET34_IMAGENET = f'{KEYS.RESNET34}/{KEYS.IMAGENET}'
KEYS.RESNET50_IMAGENET = f'{KEYS.RESNET50}/{KEYS.IMAGENET}'
KEYS.RESNET101_IMAGENET = f'{KEYS.RESNET101}/{KEYS.IMAGENET}'
KEYS.RESNET152_IMAGENET = f'{KEYS.RESNET152}/{KEYS.IMAGENET}'
KEYS.SHUFFLENET_V2_X1_0_IMAGENET = f'{KEYS.SHUFFLENET_V2_X1_0}/{KEYS.IMAGENET}'
KEYS.SQUEEZENET1_0_IMAGENET = f'{KEYS.SQUEEZENET1_0}/{KEYS.IMAGENET}'
KEYS.VGG11_IMAGENET = f'{KEYS.VGG11}/{KEYS.IMAGENET}'
KEYS.VGG13_IMAGENET = f'{KEYS.VGG13}/{KEYS.IMAGENET}'
KEYS.VGG16_IMAGENET = f'{KEYS.VGG16}/{KEYS.IMAGENET}'
KEYS.VGG19_IMAGENET = f'{KEYS.VGG19}/{KEYS.IMAGENET}'

KEYS.ALEXNET_PLACES365 = f'{KEYS.ALEXNET}/{KEYS.PLACES365}'
KEYS.RESNET18_PLACES365 = f'{KEYS.RESNET18}/{KEYS.PLACES365}'
KEYS.RESNET152_PLACES365 = f'{KEYS.RESNET152}/{KEYS.PLACES365}'
KEYS.BIGGAN_PLACES365 = f'{KEYS.BIGGAN}/{KEYS.PLACES365}'

KEYS.ALEXNET_IMAGENET_BLURRED = f'{KEYS.ALEXNET}/{KEYS.IMAGENET_BLURRED}'
KEYS.DENSENET121_IMAGENET_BLURRED = (f'{KEYS.DENSENET121}/'
                                     f'{KEYS.IMAGENET_BLURRED}')
KEYS.DENSENET201_IMAGENET_BLURRED = (f'{KEYS.DENSENET201}/'
                                     f'{KEYS.IMAGENET_BLURRED}')
KEYS.RESNET18_IMAGENET_BLURRED = f'{KEYS.RESNET18}/{KEYS.IMAGENET_BLURRED}'
KEYS.RESNET34_IMAGENET_BLURRED = f'{KEYS.RESNET34}/{KEYS.IMAGENET_BLURRED}'
KEYS.RESNET50_IMAGENET_BLURRED = f'{KEYS.RESNET50}/{KEYS.IMAGENET_BLURRED}'
KEYS.RESNET101_IMAGENET_BLURRED = f'{KEYS.RESNET101}/{KEYS.IMAGENET_BLURRED}'
KEYS.RESNET152_IMAGENET_BLURRED = f'{KEYS.RESNET152}/{KEYS.IMAGENET_BLURRED}'
KEYS.VGG11_IMAGENET_BLURRED = f'{KEYS.VGG11}/{KEYS.IMAGENET_BLURRED}'
KEYS.VGG13_IMAGENET_BLURRED = f'{KEYS.VGG13}/{KEYS.IMAGENET_BLURRED}'
KEYS.VGG16_IMAGENET_BLURRED = f'{KEYS.VGG16}/{KEYS.IMAGENET_BLURRED}'
KEYS.VGG19_IMAGENET_BLURRED = f'{KEYS.VGG19}/{KEYS.IMAGENET_BLURRED}'
KEYS.MOBILENET_V2_IMAGENET_BLURRED = (f'{KEYS.MOBILENET_V2}/'
                                      f'{KEYS.IMAGENET_BLURRED}')
KEYS.SHUFFLENET_V2_X1_0_IMAGENET_BLURRED = (f'{KEYS.SHUFFLENET_V2_X1_0}/'
                                            f'{KEYS.IMAGENET_BLURRED}')
KEYS.SQUEEZENET1_0_IMAGENET_BLURRED = (f'{KEYS.SQUEEZENET1_0}/'
                                       f'{KEYS.IMAGENET_BLURRED}')

KEYS.GENERATORS = 'gen'
KEYS.CLASSIFIERS = 'cls'
KEYS.BASE = 'base'
KEYS.NOT_ALEXNET_IMAGENET = f'not-{KEYS.ALEXNET}-{KEYS.IMAGENET}'
KEYS.NOT_ALEXNET_PLACES365 = f'not-{KEYS.ALEXNET}-{KEYS.PLACES365}'
KEYS.NOT_RESNET152_IMAGENET = f'not-{KEYS.RESNET152}-{KEYS.IMAGENET}'
KEYS.NOT_RESNET152_PLACES365 = f'not-{KEYS.RESNET152}-{KEYS.PLACES365}'
KEYS.NOT_BIGGAN_IMAGENET = f'not-{KEYS.BIGGAN}-{KEYS.IMAGENET}'
KEYS.NOT_BIGGAN_PLACES365 = f'not-{KEYS.BIGGAN}-{KEYS.PLACES365}'

# We can group the datasets of neuron annotations in a bunch of interesting
# ways. Here are the most common, used throughout the project. To load a
# grouping, simply do e.g.:
# >>> import lv.zoo
# >>> group = lv.zoo.DATASET_GROUPS['gen']
# >>> dataset = lv.zoo.datasets(*group)
DATASET_GROUPINGS = {
    KEYS.BASE: (
        KEYS.ALEXNET_IMAGENET,
        KEYS.ALEXNET_PLACES365,
        KEYS.RESNET152_IMAGENET,
        KEYS.RESNET152_PLACES365,
        KEYS.BIGGAN_IMAGENET,
        KEYS.BIGGAN_PLACES365,
    ),
    KEYS.CLASSIFIERS: (
        KEYS.ALEXNET_IMAGENET,
        KEYS.ALEXNET_PLACES365,
        KEYS.RESNET152_IMAGENET,
        KEYS.RESNET152_PLACES365,
    ),
    KEYS.GENERATORS: (
        KEYS.BIGGAN_IMAGENET,
        KEYS.BIGGAN_PLACES365,
    ),
    KEYS.IMAGENET: (
        KEYS.ALEXNET_IMAGENET,
        KEYS.RESNET152_IMAGENET,
        KEYS.BIGGAN_IMAGENET,
    ),
    KEYS.PLACES365: (
        KEYS.ALEXNET_PLACES365,
        KEYS.RESNET152_PLACES365,
        KEYS.BIGGAN_PLACES365,
    ),
    KEYS.ALEXNET: (
        KEYS.ALEXNET_IMAGENET,
        KEYS.ALEXNET_PLACES365,
    ),
    KEYS.RESNET152: (
        KEYS.RESNET152_IMAGENET,
        KEYS.RESNET152_PLACES365,
    ),
    KEYS.BIGGAN: (
        KEYS.BIGGAN_IMAGENET,
        KEYS.BIGGAN_PLACES365,
    ),
    KEYS.NOT_ALEXNET_IMAGENET: (
        KEYS.ALEXNET_PLACES365,
        KEYS.RESNET152_IMAGENET,
        KEYS.RESNET152_PLACES365,
        KEYS.BIGGAN_IMAGENET,
        KEYS.BIGGAN_PLACES365,
    ),
    KEYS.NOT_ALEXNET_PLACES365: (
        KEYS.ALEXNET_IMAGENET,
        KEYS.RESNET152_IMAGENET,
        KEYS.RESNET152_PLACES365,
        KEYS.BIGGAN_IMAGENET,
        KEYS.BIGGAN_PLACES365,
    ),
    KEYS.NOT_RESNET152_IMAGENET: (
        KEYS.ALEXNET_IMAGENET,
        KEYS.ALEXNET_PLACES365,
        KEYS.RESNET152_PLACES365,
        KEYS.BIGGAN_IMAGENET,
        KEYS.BIGGAN_PLACES365,
    ),
    KEYS.NOT_RESNET152_PLACES365: (
        KEYS.ALEXNET_IMAGENET,
        KEYS.ALEXNET_PLACES365,
        KEYS.RESNET152_IMAGENET,
        KEYS.BIGGAN_IMAGENET,
        KEYS.BIGGAN_PLACES365,
    ),
    KEYS.NOT_BIGGAN_IMAGENET: (
        KEYS.ALEXNET_IMAGENET,
        KEYS.ALEXNET_PLACES365,
        KEYS.RESNET152_IMAGENET,
        KEYS.RESNET152_PLACES365,
        KEYS.BIGGAN_PLACES365,
    ),
    KEYS.NOT_BIGGAN_PLACES365: (
        KEYS.ALEXNET_IMAGENET,
        KEYS.ALEXNET_PLACES365,
        KEYS.RESNET152_IMAGENET,
        KEYS.RESNET152_PLACES365,
        KEYS.BIGGAN_IMAGENET,
    ),
}


def hub() -> hubs.DatasetHub:
    """Return all dataset configs."""
    configs = {}

    # Configs for annotated models.
    for key in (KEYS.ALEXNET_IMAGENET, KEYS.ALEXNET_PLACES365,
                KEYS.BIGGAN_IMAGENET, KEYS.BIGGAN_PLACES365,
                KEYS.DINO_VITS8_IMAGENET, KEYS.RESNET152_IMAGENET,
                KEYS.RESNET152_PLACES365):
        configs[key] = hubs.DatasetConfig(
            datasets.AnnotatedTopImagesDataset,
            url=f'{hubs.HOST}/data/{key.replace("/", "-")}.zip',
            annotation_count=3)

    # Extra configs for models that have blurred-imagenet versopns.
    for model in (KEYS.ALEXNET, KEYS.RESNET152):
        key = KEYS[f'{model.upper()}_IMAGENET_BLURRED']
        configs[key] = hubs.DatasetConfig(datasets.TopImagesDataset)

    # Extra configs for models that have places365 versions.
    for model in (KEYS.RESNET18,):
        key = KEYS[f'{model.upper()}_PLACES365']
        configs[key] = hubs.DatasetConfig(datasets.TopImagesDataset)

    # Configs for all other models that have both imagenet/blurred-imagenet
    # versions available.
    for model in (KEYS.DENSENET121, KEYS.DENSENET201, KEYS.MOBILENET_V2,
                  KEYS.RESNET18, KEYS.RESNET34, KEYS.RESNET50, KEYS.RESNET101,
                  KEYS.SHUFFLENET_V2_X1_0, KEYS.SQUEEZENET1_0, KEYS.VGG11,
                  KEYS.VGG13, KEYS.VGG16, KEYS.VGG19):
        for dataset in (KEYS.IMAGENET, KEYS.IMAGENET_BLURRED):
            key = KEYS[f'{model.upper()}_{dataset.upper().replace("-", "_")}']
            configs[key] = hubs.DatasetConfig(datasets.TopImagesDataset)

    return hubs.DatasetHub(**configs)


def load(name: str = 'all', **kwargs: Any) -> torch.utils.data.Dataset:
    """Load some or all of MILANNOTATIONS.

    Args:
        name (str): Name of specific model to load top images and/or
            annotations for, or name of a group of models
            (see DATASET_GROUPINGS) to load. Defaults to all annotated models
            used for training MILAN.

    Returns:
        torch.utils.data.Dataset: The loaded dataset.

    """
    if name in DATASET_GROUPINGS:
        dataset = hub().load_all(*DATASET_GROUPINGS[name], **kwargs)
    elif name in KEYS:
        dataset = hub().load(name, **kwargs)
        assert isinstance(
            dataset,
            (datasets.TopImagesDataset, datasets.AnnotatedTopImagesDataset),
        ), dataset
    else:
        raise KeyError(f'unknown milannotations set: {name}')
    return dataset
