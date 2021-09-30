"""Dissect a pretrained vision model."""
import argparse
import pathlib

from lv.dissection import dissect, zoo
from lv.utils import env

from torch import cuda

parser = argparse.ArgumentParser(description='dissect a vision model')
parser.add_argument('model', help='model architecture')
parser.add_argument('dataset', help='dataset model is trained on')
parser_ex = parser.add_mutually_exclusive_group()
parser_ex.add_argument('--layer-names',
                       nargs='+',
                       help='layer names to dissect')
parser_ex.add_argument(
    '--layer-indices',
    type=int,
    nargs='+',
    help='layer indices to dissect; cannot be used with --layers')
parser.add_argument('--units',
                    type=int,
                    help='only dissect the first n units (default: all)')
parser.add_argument('--results-root',
                    type=pathlib.Path,
                    help='dissection results root '
                    '(default: <project results dir> / dissection)')
parser.add_argument('--viz-root',
                    type=pathlib.Path,
                    help='dissection visualization root '
                    '(default: <project results dir> / dissection / viz)')
parser.add_argument('--model-file',
                    type=pathlib.Path,
                    help='path to model weights')
parser.add_argument('--dataset-path',
                    type=pathlib.Path,
                    help='path to dataset')
parser.add_argument('--no-viz',
                    action='store_true',
                    help='do not compute visualization')
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
# TODO(evandez): Yuck, push this into config.
elif dataset == zoo.KEYS.IMAGENET_BLURRED:
    dataset = zoo.KEYS.IMAGENET

dataset = zoo.dataset(dataset, path=args.dataset_path)

if args.layer_names:
    layers = args.layer_names
elif args.layer_indices:
    layers = [layers[index] for index in args.layer_indices]
assert layers is not None, 'should always be >= 1 layer'

units = None
if args.units:
    units = range(args.units)

results_root = args.results_root
if results_root is None:
    results_root = env.results_dir() / 'dissection'
results_dir = results_root / args.model / args.dataset

viz_root = args.viz_root
viz_dir = None
if viz_root is not None:
    viz_dir = viz_root / args.model / args.dataset
elif not args.no_viz:
    viz_dir = results_root / 'viz' / args.model / args.dataset

for layer in layers:
    if generative:
        dissect.generative(model,
                           dataset,
                           layer=layer,
                           units=units,
                           results_dir=results_dir,
                           viz_dir=viz_dir,
                           save_viz=not args.no_viz,
                           device=device,
                           **config.dissection.kwargs)
    else:
        dissect.discriminative(model,
                               dataset,
                               layer=layer,
                               units=units,
                               results_dir=results_dir,
                               viz_dir=viz_dir,
                               save_viz=not args.no_viz,
                               device=device,
                               **config.dissection.kwargs)
