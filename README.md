# Describing Neurons in Neural Networks

[**Natural Language Descriptions of Deep Visual Features**](https://arxiv.org/abs/2201.11114)<br>
[Evan Hernandez](https://evandez.com), [Sarah Schwettmann](https://cogconfluence.com), [David Bau](http://davidbau.com), [Teona Bagashvili](https://sites.allegheny.edu/admissions/teona-bagashvili/), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Jacob Andreas](https://www.mit.edu/~jda/).<br>
ICLR 2022 (oral).<br>

<hr>

In this paper, we ask: what concepts are encoded in the features of neural networks? One way to answer this question is to look at individual neurons in the network: do these individual units detect interesting concepts on their own?

We can represent the behavior of a neuron by the set of inputs that cause it to activate most strongly, a set we name the "top-activating exemplars"--in the case of networks trained on computer vision tasks, the exemplars take the form of image regions. While they highlight that the corresponding neurons are sensitive to interesting perceptual-, object-, scene-level concepts, it is challenging to analyze the exemplars for all of the neurons at scale. Previous automated labeling techniques pulled candidate labels from a fixed, pre-specified set, limiting the kinds of behaviors they could surface.

This project introduces **MILAN**, an approach for generating natural language descriptions of neurons in neural networks. MILAN is built on another set of neural networks, which in turned are trained on a novel dataset of (image regions, description) pairs called **MILANNOTATIONS**. The descriptions are chosen so that they have high pointwise mutual information with the neuron, encouraging them to be both truthful and specific.

![MILAN overview](/www/milan-overview.png)

In our paper, we show that MILAN reliably generates useful descriptions for neurons in new, unseen vision networks. We also demonstrate how MILAN can be applied to a number of downstream interpretability tasks, such as analyzing feature importance, auditing models for unexpected features, and editing spurious correlations out of models.

## Setup

All code is tested on `MacOS Monterey (>= 12.2.1)` and `Ubuntu 20.04` using `Python >= 3.8`. It may run in other environments, but because it uses a lot of newer Python features, we make no guarantees.

To run the code, set up a virtual environment and install the dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.in
spacy download en_core_web_sm
```

To validate that everything works, run the presubmit script, which in turn performs type checking, linting and unit testing:
```bash
./presubmit.sh
```

Finally, to control where data and models are downloaded and where results are written, you can set the following environment variables (which have the defaults below):
```bash
MILAN_DATA_DIR=./data
MILAN_MODELS_DIR=./models
MILAN_RESULTS_DIR=./results
```

## Downloading MILANNOTATIONS

We collected over 50k human descriptions of sets of image regions, which were taken from the top-activating images of several base models. We make the full set of annotations and top-image masks publicly available.

For legal reasons, we cannot release the raw source images from ImageNet, but we include pointers to the images as they appear in a standard [`ImageFolder`](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder). If you use the library described further down, it will automatically import your locally downloaded copy of ImageNet with MILANNOTATIONS.

The table below details the annotated base models.

| Model | Task | # Units | # Desc. | Source Images | Download |
|-------|------|---------|---------|---------------|----------|
| alexnet/imagenet | class | 1k | 3k | [request access](https://www.image-net.org) | [zip](http://milan.csail.mit.edu/data/alexnet-imagenet.zip) |
| alexnet/places365 | class | 1k | 3k | included in zip | [zip](http://milan.csail.mit.edu/data/alexnet-places365.zip) |
| resnet152/imagenet | class | 3k | 9k | [request access](https://www.image-net.org) | [zip](http://milan.csail.mit.edu/data/resnet152-imagenet.zip) |
| resnet152/places365 | class | 4k | 12k | included in zip | [zip](http://milan.csail.mit.edu/data/resnet152-places365.zip)
biggan/imagenet | gen | 4k | 12k | included in zip | [zip](http://milan.csail.mit.edu/data/biggan-imagenet.zip)
biggan/places365 | gen | 4k | 12k | included in zip | [zip](http://milan.csail.mit.edu/data/biggan-places365.zip)
dino_vits8/imagenet | BYOL | 1.2k | 3.6k | [request access](https://www.image-net.org) | [zip](http://milan.csail.mit.edu/data/dino_vits8-imagenet.zip)

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

## Downloading MILAN

We offer several pretrained MILAN models trained on different subsets of MILANNOTATIONS:

| Version | Trained On | Download |
|---------|------------|----------|
| base | {alexnet, resnet152, biggan} x {imagenet, places365} | [weights](http://milan.csail.mit.edu/models/milan-base.pth)
| cls | {alexnet, resnet152} x {imagenet, places365} | [weights](http://milan.csail.mit.edu/models/milan-cls.pth)
| gen | {biggan} x {imagenet, places365} | [weights](http://milan.csail.mit.edu/models/milan-gen.pth)
| imagenet | {alexnet, resnet152, biggan} x {imagenet} | [weights](http://milan.csail.mit.edu/models/milan-imagenet.pth)
| places365 | {alexnet, resnet152, biggan} x {places365} | [weights](http://milan.csail.mit.edu/models/milan-places365.pth)
| alexnet | {alexnet} x {imagenet, places365} | [weights](http://milan.csail.mit.edu/models/milan-alexnet.pth)
| resnet152 | {resnet152} x {imagenet, places365} | [weights](http://milan.csail.mit.edu/models/milan-resnet152.pth)
| biggan | {biggan} x {imagenet, places365} | [weights](http://milan.csail.mit.edu/models/milan-biggan.pth)

The root module for MILAN is `Decoder` inside [src.milan.decoders](src/milan/decoders.py). However, you should not have to interact with it because the library will automatically download and configure the model for you. Here is a minimal usage example applied to DINO:

```python
from src import milan, milannotations

# Load the base model trained on all available data (except ViT):
decoder = milan.pretrained('base')

# Load some neurons to describe; we'll use unit 10 in layer 9.
dataset = milannotations.load('dino_vits8/imagenet')
sample = dataset.lookup('blocks.9.mlp.fc1', 10)

# Caption the top images.
outputs = decoder(sample.images[None], masks=sample.masks[None])
print(outputs.captions[0])
```

**New, April 2022**: Add `+clip` to any of the keys above (e.g. `base+clip`) to augment the MILAN decoder with CLIP. This works by first sampling candidate descriptions from MILAN, and then reranking them with CLIP. This approach was not evaluated in the original paper, but  qualitatively produces more detailed, if less fluent, descriptions.

## Applying MILAN to new models

Do you want to get neuron descriptions for your own model? Our library makes very few assumptions about the model to be described, other than (1) it has a PyTorch implementation, (2) the inputs or outputs of the model are images, and (3) the layer containing the target neurons corresponds directly to a torch module so its output can be hooked during exemplar construction.

To get started, follow the next three steps.

### Step 1: Configure your model

MILAN first needs to know:
- how to load your model
- how to load the dataset containing source images
- what layers look in for neurons

This information is specified inside [src/exemplars/models.py](src/exemplars/models.py) and [src/exemplars/datasets.py](src/exemplars/datasets.py) using the `ModelConfig` and `DatasetConfig` constructs, respectively. Simply add configs for your model and dataset inside `default_model_configs` and `default_dataset_configs`.

To illustrate how the configs work, here is a model config for one of the models used in the original paper:

```python
def default_model_configs(...):
  configs = {
    ...
    'resnet18/imagenet': ModelConfig(
      torchvision.models.resnet18,
      pretrained=True,
      load_weights=False,
      layers=('conv1', 'layer1', 'layer2', 'layer3', 'layer4'),
    )
    ...
  }
```
Walking through each of the pieces:
- **`torchvision.models.resnet18`**: A function that returns a torch module.
- **`pretrained`**: This argument is unrecognized by `ModelConfig`, so it will be forwarded to the factory function when it is called. In this case, it is used by torchvision to signal that we want to download the pretrained model.
- **`load_weights`**: When this is true, the config will look for a file containing pretrained weights under `$MILAN_MODELS_DIR/resnet18-imagenet.pth` and try to load them into the model. Since torchvision downloads the weights for us, we set this to False.
- **`layers`**: A sequence of fully specified paths to the layers you want to compute exemplars for. E.g., if you specifically want to use the first conv layer in the first sub-block of layer1 of resnet18, you would specify it as `layer1.0.conv1`.

The dataset configs behave similarly. See the class definitions in [src/utils/hubs.py](src/utils/hubs.py) for a full list of options.

### Step 2: Compute top-activating images

Once you've configured your model, you can run the script below to compute exemplars for your model. Continuing our ResNet18 example:
```bash
python3 -m scripts.compute_exemplars resnet18 imagenet --device cuda
```
This will write the top images under `$MILAN_RESULTS_DIR/exemplars/resnet18/imagenet` and link it to the corresponding directory in `$MILAN_DATA_DIR` so you can load it with the MILANNOTATIONS library.

### Step 3: Decode Descriptions with MILAN

Finally, you can use one of the pretrained MILAN models to get descriptions for the exemplars you just computed. As before, just call a script:

```bash
python3 -m scripts.compute_milan_descriptions resnet18 imagenet --device cuda
```

This will write the descriptions to a CSV in `$MILAN_RESULTS_DIR/descriptions/resnet18/imagenet`.

## Running experiments

All experiments from the main paper can be reproduced using scripts in the [experiments](experiments) subdirectory. Here is an example of how to invoke these scripts with the correct `PYTHONPATH`:
```bash
python3 -m experiments.generalization --experiments within-network --precompute-features --device cuda
```

A myriad of other scripts can be found under the [scripts](scripts) directory. These do not correspond to any particular experiment, but are used for more general or miscellaneous tasks such as training MILAN, cleaning AMT data, and generating visualizations. For a full description of how to use a script, use the help (`-h`) flag.


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
