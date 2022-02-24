# Describing Neurons in Neural Networks

## DISCLAIMER: Under active changes! Things will break!

![MILAN overview](/www/milan-overview.png)

## Setup

All code is tested on `MacOS Monterey (>= 12.2.1)` and `Ubuntu 20.04` using `Python >= 3.8`. It may run in other environments, but because it uses a lot of newer Python features, we make no guarantees.

To run the code, set up a virtual environment and install the dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

To validate that everything works, run the presubmit script, which in turn performs typing checking, linting and unit testing:
```bash
./presubmit.sh
```

Finally, to control where data and models are downloaded and where results are written, you can set the following environment variables (which have the defaults below):
```bash
MILAN_DATA_DIR=./data
MILAN_MODELS_DIR=./models
MILAN_RESULTS_DIR=./results
```

## Using `MILANNOTATIONS`

We collected over 50k human descriptions of sets of image regions, which were taken from the top-activating images of several base models. We make the full set of annotations and top-image masks publicly available.

For legal reasons, we cannot release the raw source images from ImageNet, but we include pointers to the images as they appear in a standard [`ImageFolder`](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder). If you use the library described further down, it will automatically import your locally downloaded copy of ImageNet with MILANNOTATIONS.

The table below details the annotated base models.

| Model | Task | # Units | # Desc. | Source Images | Download |
|-------|------|---------|---------|---------------|----------|
| alexnet/imagenet | class | 1k | 3k | [request access](https://www.image-net.org) | [zip](https://milan.csail.mit.edu/data/alexnet-imagenet.zip) |
| alexnet/places365 | class | 1k | 3k | included in zip | [zip](https://milan.csail.mit.edu/data/alexnet-places365.zip) |
| resnet152/imagenet | class | 3k | 9k | [request access](https://www.image-net.org) | [zip](https://milan.csail.mit.edu/data/resnet152-imagenet.zip) |
| resnet152/places365 | class | 4k | 12k | included in zip | [zip](https://milan.csail.mit.edu/data/resnet152-places365.zip)
biggan/imagenet | gen | 4k | 12k | included in zip | [zip](https://milan.csail.mit.edu/data/biggan-imagenet.zip)
biggan/places365 | gen | 4k | 12k | included in zip | [zip](https://milan.csail.mit.edu/data/biggan-places365.zip)
dino_vits8/imagenet | BYOL | 1.2k | 3.6k | [request access](https://www.image-net.org) | [zip](https://milan.csail.mit.edu/data/dino_vits8-imagenet.zip)

<!--
We also provide precomputed exemplars for other image classification models analyzed in the original paper. They are all based on ImageNet and require you to have a local copy (i.e., `$MILAN_DATA_DIR/imagenet/val` should exist):

| Model | # Units | Download |
|-------|---------|----------|
alexnet/imagenet-blurred | |

-->
We provide a fully featured library for downloading and using this data. Here are some examples:

```python
from src import milannotations

# Load all training data (AlexNet/ResNet152/BigGAN on ImageNet/Places):
base = milannotations.load('base')

# Load annotations for all imagenet models:
gen = milannotations.load('imagenet')

# Load annotations for a specific model:
alexnet_imagenet = milannotations.load('alexnet/imagenet')
resnet_imagenet = milannotations.load('resnet152/imagenet')
```

For a complete demo on interacting with `MILANNOTATIONS`, see
[notebooks/milannotations.ipynb](notebooks/milannotations.ipynb).

## Using `MILAN`

We offer several pretrained MILAN models trained on different subsets of `MILANNOTATIONS`. The library automatically downloads and configures the models. Here is a minimal usage example applied to DINO:

```python
from src import milan, milannotations

# Load the base model trained on all available data (except ViT):
decoder = milan.pretrained('base')

# Load some neurons to describe; we'll use unit 10 in layer 9.
dataset = milannotations.load('dino_vits8/imagenet')
sample = dataset.lookup('blocks.9.mlp.fc1', 10)

# Caption the top images.
outputs = milan(sample.images, masks=sample.masks)
print(outputs.captions[0])
```

If you would like to apply `MILAN` to a new model, you must first compute the top-activating images for its neurons. You can do this in two steps. First, add a config to [src/exemplars/models.py](src/exemplars/models.py) for your model and to [src/exemplars/datasets.py](src/exemplars/datasets.py) for the dataset you want the top images to be taken from. Then, you can call the following script:
```bash
python3 -m scripts.compute_exemplars your_model_name your_dataset_name --device cuda
```
Please note that, by default, this script will:
- look for your model under `$MILAN_MODELS_DIR/your_model_name.pth`;
- look for your dataset under `$MILAN_DATA_DIR/your_dataset_name`;
- write top images to `$MILAN_RESULTS_DIR/your_model_name/your_dataset_name`;
- link the directory above to an equivalent directory in `$MILAN_DATA_DIR`.

<!-- For a more detailed demo of `MILAN`'s features, see [notebooks/milan.ipynb](notebooks/milan.ipynb). -->

## Running experiments & other scripts

All experiments from the main paper can be reproduced using scripts in the [experiments](experiments) subdirectory. Here is an example of how to invoke these scripts with the correct `PYTHONPATH`:
```bash
python3 -m experiments.generalization --experiments within-network --precompute-features --device cuda
```

A myriad of other scripts can be found under the [scripts](scripts) directory. These do not correspond to any particular experiment, but are used for more general or miscellaneous tasks such as training `MILAN`, cleaning AMT data, and generating visualizations. For a full description of how to use a script, use the help (`-h`) flag.


## Contributing

While this library is not designed for industrial use (it's just a research project), we do believe research code should support reproducibility.  If you have issues running our code in the supported environment, please open an issue on this repository.

If you find ways to improve our code, you may also submit a pull request. Before doing so, please ensure that the code type checks, lints cleanly, and passes all unit tests. The following command should produce green text:
```bash
./presubmit.sh
```

## Bibtex

```latex
@InProceedings{hernandez2022natural,
  title     =   {Natural Language Descriptions of Deep Visual Features},
  author    =   {Hernandez, Evan and Schwettmann, Sarah and Bau, David and Bagashvili, Teona, and Torralba, Antonio and Andreas, Jacob},
  booktitle =   {International Conference on Learning Representations},
  year      =   {2022},
  url       =   {https://arxiv.org/abs/2201.11114}
}
```
