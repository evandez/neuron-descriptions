"""Dissect a pretrained vision model."""
import argparse
import pathlib

from lv.dissection import dissect, zoo
from lv.utils import env

from torch import cuda

parser = argparse.ArgumentParser(description='dissect a vision model')
parser.add_argument('model', help='model architecture')
parser.add_argument('dataset', help='dataset model is trained on')
parser.add_argument('--layers', nargs='+', help='layers to dissect')
parser.add_argument('--results-root',
                    type=pathlib.Path,
                    help='dissection results root '
                    '(default: <project results dir> / dissection)')
parser.add_argument('--viz-root',
                    type=pathlib.Path,
                    help='dissection visualization root '
                    '(default: <results root> / viz)')
parser.add_argument('--model-file',
                    type=pathlib.Path,
                    help='path to model weights')
parser.add_argument('--dataset-path',
                    type=pathlib.Path,
                    help='path to dataset')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

model, layers, config = zoo.model(args.model,
                                  args.dataset,
                                  map_location=device,
                                  path=args.model_file)

dataset, generative = args.dataset, False
if isinstance(config.dissection, zoo.GenerativeModelDissectionConfig):
    dataset = config.dissection.dataset
    generative = True

dataset = zoo.dataset(dataset, path=args.dataset_path)

layers = args.layers or layers
assert layers is not None, 'should always be >= 1 layer'

results_root = args.results_root
if results_root is None:
    results_root = env.results_dir() / 'dissection'
results_dir = results_root / args.model / args.dataset

viz_root = args.viz_root
if viz_root is None:
    viz_root = results_root / 'viz'
viz_dir = viz_root / args.model / args.dataset

for layer in layers:
    if generative:
        dissect.generative(model,
                           dataset,
                           layer=layer,
                           results_dir=results_dir,
                           viz_dir=viz_dir,
                           device=device,
                           **config.dissection.kwargs)
    else:
        dissect.discriminative(model,
                               dataset,
                               layer=layer,
                               results_dir=results_dir,
                               viz_dir=viz_dir,
                               device=device,
                               **config.dissection.kwargs)
