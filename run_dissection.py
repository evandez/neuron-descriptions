"""Dissect a pretrained vision model."""
import argparse
import pathlib

from lv.dissection import dissect, zoo

import torch

parser = argparse.ArgumentParser(description='dissect a vision model')
parser.add_argument('model', help='model architecture')
parser.add_argument('dataset', help='dataset model is trained on')
parser.add_argument('--layers', nargs='+', help='layers to dissect')
parser.add_argument('--results-dir',
                    type=pathlib.Path,
                    default='.dissection',
                    help='directory to write dissection results')
parser.add_argument('--model-file',
                    type=pathlib.Path,
                    help='path to model weights')
parser.add_argument('--dataset-path',
                    type=pathlib.Path,
                    help='path to dataset')
parser.add_argument('--cuda', action='store_true', help='use cuda')
args = parser.parse_args()

device = torch.device('cuda' if args.cuda else 'cpu')

model, layers, config = zoo.model(args.model,
                                  args.dataset,
                                  map_location=device,
                                  path=args.model_file)
generative = config.dissection.generative
kwargs = config.dissection.kwargs

# TODO(evandez): Yuck, think of a better way to do this.
dataset = zoo.dataset(
    f'{args.model}-zs-{args.dataset}' if generative else args.dataset,
    path=args.dataset_path)

layers = args.layers or layers
assert layers is not None, 'should always be >= 1 layer'

for layer in layers:
    results_dir = args.results_dir / args.model / args.dataset
    if generative:
        dissect.generative(model,
                           dataset,
                           layer=layer,
                           results_dir=results_dir,
                           device=device,
                           **kwargs)
    else:
        dissect.sequential(model,
                           dataset,
                           layer=layer,
                           results_dir=results_dir,
                           device=device,
                           **kwargs)
