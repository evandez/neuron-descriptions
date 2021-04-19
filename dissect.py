"""Dissect a pretrained vision model."""
import argparse
import pathlib

from lv.dissection import dissect, zoo
from third_party.netdissect import pbar

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
parser.add_argument('--dataset-dir',
                    type=pathlib.Path,
                    help='path to dataset root')
parser.add_argument('--cuda', action='store_true', help='use cuda')
args = parser.parse_args()

device = torch.device('cuda' if args.cuda else 'cpu')

model, layers, config = zoo.model(args.model,
                                  args.dataset,
                                  map_location=device,
                                  path=args.model_file)
dataset = zoo.dataset(args.dataset, path=args.dataset_dir)

layers = args.layers or layers
assert layers is not None, 'should always be >= 1 layer'

dissect_fn = dissect.generative if config.generative else dissect.sequential

for layer in layers:
    pbar.post(layer=layer)
    dissect_fn(  # type: ignore
        model,
        dataset,
        layer=layer,
        device=device,
        results_dir=args.results_dir / args.model / args.dataset)
