"""Run CompExp baseline.

Reimplemented from https://github.com/jayelm/compexp because (1) the original
repository does not work with generative models, and (2) it uses ground-truth
image segmentations from the ADE20k dataset, instead of constructing them
on the fly from a segmentation model like NetDissect does.
"""
import argparse
import pathlib
import shutil
from typing import Any, Mapping

from lv.deps.netdissect import (nethook, pbar, renormalize, segmenter, tally,
                                upsample)
from lv.dissection import dissect, transforms, zoo
from lv.utils import env

import torch
import torchvision.transforms
from torch import cuda

parser = argparse.ArgumentParser(description='run compexp baseline')
parser.add_argument('model', help='model to dissect')
parser.add_argument('dataset', help='dataset model was trained on')
parser.add_argument('--layers',
                    nargs='+',
                    help='layers to dissect (default: all)')
parser.add_argument(
    '--quantile',
    type=float,
    default=.995,
    help='quantile to use for activation threshold (default: .995)')
parser.add_argument('--model-file',
                    type=pathlib.Path,
                    help='model weight file (default: None)')
parser.add_argument('--dataset-path',
                    type=pathlib.Path,
                    help='path to dataset (default: None)')
parser.add_argument(
    '--models-dir',
    type=pathlib.Path,
    help='directory to store cache files in (default: project models dir)')
parser.add_argument('--results-dir',
                    type=pathlib.Path,
                    help='root dir for intermediate and final results '
                    '(default: project results dir)')
parser.add_argument('--clear-results-dir',
                    action='store_true',
                    help='if set, clear results dir (default: do not)')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    help='image batch size (default: 128)')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

# Prepare result dir.
results_dir = args.results_dir or (env.results_dir() / 'compexp')
if args.clear_results_dir and results_dir.exists():
    shutil.rmtree(results_dir)
results_dir.mkdir(exist_ok=True, parents=True)

# Load the model to dissect and the dataset to dissect on.
model, layers, config = zoo.model(args.model,
                                  args.dataset,
                                  map_location=device)
layers = args.layers or layers

dataset, generative = args.dataset, False
if isinstance(config.dissection, zoo.GenerativeModelDissectionConfig):
    dataset = config.dissection.dataset
    generative = True

# TODO(evandez): YUCK! Need to commonize this somewhere.
kwargs = {}
if args.model == zoo.KEYS.ALEXNET and args.datset == zoo.KEYS.PLACES365:
    kwargs['transform'] = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(227),
        torchvision.transforms.ToTensor(),
        renormalize.NORMALIZER['imagenet'],
    ])

dataset = zoo.dataset(dataset, path=args.dataset_path, **kwargs)

# Load the segmentation model for later; its path must be hardcoded because
# of how the library is written.
models_dir = args.models_dir or env.models_dir()
segmodel_cache_dir = pathlib.Path('datasets/segmeodel')
segmodel_cache_dir.mkdir(exist_ok=True, parents=True)
segmenter.ensure_segmenter_downloaded(str(segmodel_cache_dir), 'color')
segmodel = segmenter.MergedSegmenter([
    segmenter.UnifiedParsingSegmenter(all_parts=True, segdiv='quad'),
    segmenter.SemanticSegmenter(segvocab='color',
                                segarch=('resnet18dilated', 'ppm_deepsup')),
])
segcatlabels = segmodel.get_label_and_category_names()[0]
seglabels = [label for label, _ in segcatlabels]
segrenorm = renormalize.renormalizer(source='zc' if generative else 'imagenet',
                                     target='zc')

# We'll do this layer by layer because of memory constraints.
for layer in layers:
    layer = str(layer)

    cache_key = f'{args.model}_{args.dataset}_{layer}'

    # Begin by computing activation statistics.
    tally_cache_file = results_dir / f'{cache_key}_tally.npz'
    if generative:
        _, rq = dissect.generative(model,
                                   dataset,
                                   layer=layer,
                                   device=device,
                                   batch_size=args.batch_size,
                                   save_results=False,
                                   save_viz=False,
                                   tally_cache_file=tally_cache_file,
                                   **config.dissection.kwargs)
    else:
        _, rq = dissect.discriminative(model,
                                       dataset,
                                       layer=layer,
                                       device=device,
                                       batch_size=args.batch_size,
                                       save_results=False,
                                       save_viz=False,
                                       tally_cache_file=tally_cache_file,
                                       **config.dissection.kwargs)
    levels = rq.quantiles(args.quantile).reshape(1, -1, 1, 1).to(device)

    # Compute unit masks and "ground truth" segmentation masks for every
    # image in the dataset.
    pbar.descnext('compute seg/unit masks')
    with nethook.InstrumentedModel(model) as instr:
        instr.retain_layer(layer, detach=False)
        upsampler = None

        def compute_segs_and_unit_masks(
                *inputs: Any) -> Mapping[str, torch.Tensor]:
            """Segment the image batch."""
            inputs = transforms.map_location(inputs, device)

            # If our model is generative: the images come from the model, not
            # from the data batch.
            if generative:
                kwargs = config.dissection.kwargs
                transform_inputs = kwargs.get('transform_inputs',
                                              transforms.identities)
                transform_hiddens = kwargs.get('transform_hiddens',
                                               transforms.identity)
                transform_outputs = kwargs.get('transform_outputs',
                                               transforms.identity)

                with torch.no_grad():
                    inputs = transform_inputs(*inputs)
                    outputs = instr(*inputs)
                    images = transform_outputs(outputs)
                    acts = transform_hiddens(instr.retained_layer(layer))

            # If our model is discriminative: the images come from the dataset,
            # and we just need to compute activations.
            else:
                kwargs = config.dissection.kwargs
                transform_inputs = kwargs.get('transform_inputs',
                                              transforms.first)
                transform_outputs = kwargs.get('transform_outputs',
                                               transforms.identity)

                images, *_ = inputs  # Just assume images are first...
                with torch.no_grad():
                    inputs = transform_inputs(*inputs)
                    instr(*inputs)
                    acts = transform_outputs(instr.retained_layer(layer))

            with torch.no_grad():
                segs = segmodel.segment_batch(segrenorm(images), downsample=4)

            global upsampler
            if upsampler is None:
                upsampler = upsample.upsampler(segs.shape[-2:],
                                               data_shape=acts.shape[-2:],
                                               image_size=images.shape[-2:])

            unit_masks = (upsampler(acts) > levels).float()

            return {
                'segs': segs,
                'unit_acts': acts,
                'unit_masks': unit_masks,
            }

        masks = tally.tally_cat_dict(compute_segs_and_unit_masks,
                                     dataset,
                                     batch_size=args.batch_size,
                                     cachefile=results_dir /
                                     f'{cache_key}_seg_unit_masks.npz')

    # Now...finally...we can do the CompExp labeling.
    # The important players are:
    #
    # `unit_masks`, 0/1 tensor
    #       of shape (n_images, n_neurons, height, width)
    #
    # `seg_masks`, int tensor
    #       of shape (n_images, n_labels_per_pixel, height, width)
    # TODO(evandez): Implement.
