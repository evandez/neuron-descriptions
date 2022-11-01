"""Dissect a pretrained vision model."""
import argparse
import pathlib

from src.exemplars import compute, datasets, models
from src.utils import env

from torch import cuda

parser = argparse.ArgumentParser(description='compute unit exemplars')
parser.add_argument('model', help='model architecture')
parser.add_argument('dataset', help='dataset of unseen examples for model')
parser_ex = parser.add_mutually_exclusive_group()
parser_ex.add_argument('--layer-names',
                       nargs='+',
                       help='layer names to compute exemplars for')
parser_ex.add_argument('--layer-indices',
                       type=int,
                       nargs='+',
                       help='layer indices to compute exemplars for; '
                       'cannot be used with --layers')
parser.add_argument(
    '--units',
    type=int,
    help='only compute exemplars for first n units (default: all)')
parser.add_argument('--data-root',
                    type=pathlib.Path,
                    help='link results (in --results-root) to this directory '
                    '(default: <project data dir> / model / dataset)')
parser.add_argument('--results-root',
                    type=pathlib.Path,
                    help='exemplars results root '
                    '(default: <project results dir> / exemplars)')
parser.add_argument('--viz-root',
                    type=pathlib.Path,
                    help='exemplars visualization root '
                    '(default: <project results dir> / exemplars / viz)')
parser.add_argument('--model-file',
                    type=pathlib.Path,
                    help='path to model weights')
parser.add_argument('--dataset-path',
                    type=pathlib.Path,
                    help='path to dataset')
parser.add_argument('--no-viz',
                    action='store_true',
                    help='do not compute visualization')
parser.add_argument('--no-link',
                    action='store_true',
                    help='do not link results to data dir')
parser.add_argument('--num-workers',
                    type=int,
                    default=16,
                    help='number of worker threads (default: 16)')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

model, layers, config = models.load(f'{args.model}/{args.dataset}',
                                    map_location=device,
                                    path=args.model_file)

dataset, generative = args.dataset, False
if isinstance(config.exemplars, models.GenerativeModelExemplarsConfig):
    dataset = config.exemplars.dataset
    generative = True
# TODO(evandez): Yuck, push this into config.
elif dataset == datasets.KEYS.IMAGENET_BLURRED:
    dataset = datasets.KEYS.IMAGENET

dataset = datasets.load(dataset, path=args.dataset_path)

if args.layer_names:
    layers = args.layer_names
elif args.layer_indices:
    layers = [layers[index] for index in args.layer_indices]
assert layers is not None, 'should always be >= 1 layer'

units = None
if args.units:
    units = range(args.units)

data_root = args.data_root
if data_root is None:
    data_root = env.data_dir()
data_dir = data_root / args.model / args.dataset

results_root = args.results_root
if results_root is None:
    results_root = env.results_dir() / 'exemplars'
results_dir = results_root / args.model / args.dataset

viz_root = args.viz_root
viz_dir = None
if viz_root is not None:
    viz_dir = viz_root / args.model / args.dataset
elif not args.no_viz:
    viz_dir = results_root / 'viz' / args.model / args.dataset

for layer in layers:
    if generative:
        compute.generative(model,
                           dataset,
                           layer=layer,
                           units=units,
                           results_dir=results_dir,
                           viz_dir=viz_dir,
                           save_viz=not args.no_viz,
                           device=device,
                           num_workers=args.num_workers,
                           **config.exemplars.kwargs)
    else:
        compute.discriminative(model,
                               dataset,
                               layer=layer,
                               units=units,
                               results_dir=results_dir,
                               viz_dir=viz_dir,
                               save_viz=not args.no_viz,
                               device=device,
                               num_workers=args.num_workers,
                               **config.exemplars.kwargs)

if not args.no_link:
    results_dir.symlink_to(data_dir, target_is_directory=True)
