"""Run CompExp baseline.

Reimplemented from https://github.com/jayelm/compexp because (1) the original
repository does not work with generative models, and (2) it uses ground-truth
image segmentations from the ADE20k dataset, instead of constructing them
on the fly from a segmentation model like NetDissect does.
"""
import argparse
import csv
import dataclasses
import functools
import multiprocessing
import pathlib
import shutil
from typing import Any, Mapping, Optional, Tuple, cast

from lv.deps.netdissect import (nethook, pbar, renormalize, segmenter, tally,
                                upsample)
from lv.dissection import dissect, transforms, zoo
from lv.utils import env

import torch
import torchvision.transforms
from torch import cuda
from tqdm.auto import tqdm


class LogicalForm:
    """Abstract class for a logical form."""

    @functools.cached_property
    def masks(self) -> torch.Tensor:
        """Return the 2D mask representing this logical form."""
        raise NotImplementedError

    def iou(self,
            others: torch.Tensor,
            total: Optional[int] = None,
            eps: float = 1e-10) -> float:
        """Compute IOU of this LF with respect to the given masks.

        Args:
            others (torch.Tensor): The other masks.
            total (Optional[int], optional): Precomputed total number of
                1-bits in the other masks. Computed by default.
            eps (float, optional): Division tolerance. Defaults to 1e-10

        Returns:
            float: IOU score.

        """
        if total is None:
            total = cast(int, others.sum().item())
        masks = self.masks
        intersection = self.masks.bitwise_and(others).sum().item()
        return intersection / (masks.sum() + total - intersection + eps)


@dataclasses.dataclass(frozen=True)
class And(LogicalForm):
    """The AND operator."""

    left: LogicalForm
    right: LogicalForm

    @functools.cached_property
    def masks(self) -> torch.Tensor:
        """Return the intersection of the two child masks."""
        return self.left.masks & self.right.masks

    def __str__(self) -> str:
        """Return the logical form as a string."""
        components = []
        for child in (self.left, self.right):
            if isinstance(child, Literal):
                components.append(str(child))
            else:
                components.append(f'({child})')
        return ' AND '.join(components)


@dataclasses.dataclass(frozen=True)
class Or(LogicalForm):
    """The OR operator."""

    left: LogicalForm
    right: LogicalForm

    @functools.cached_property
    def masks(self) -> torch.Tensor:
        """Return the union of the two child masks."""
        return self.left.masks & self.right.masks

    def __str__(self) -> str:
        """Return the logical form as a string."""
        components = []
        for child in (self.left, self.right):
            if isinstance(child, Literal):
                components.append(str(child))
            else:
                components.append(f'({child})')
        return ' OR '.join(components)


@dataclasses.dataclass(frozen=True)
class Not(LogicalForm):
    """The NOT operator."""

    term: LogicalForm

    @functools.cached_property
    def masks(self) -> torch.Tensor:
        """Return the negation of the child mask."""
        return ~self.term.masks

    def __str__(self) -> str:
        """Return the logical form as a string."""
        string = str(self.term)
        if not isinstance(self.term, Literal):
            string = f'({string})'
        return f'NOT {string}'


@dataclasses.dataclass(frozen=True)
class Literal(LogicalForm):
    """A literal term in the logical form."""

    key: int
    label: str
    masks: torch.Tensor


@dataclasses.dataclass(frozen=True)
class State:
    """Beam search state of running compexp on a unit."""

    logical_form: LogicalForm
    iou: float


@dataclasses.dataclass(frozen=True)
class Result:
    """Results for length == 1 and length <= max_lf_length."""

    unit: int
    length_1: State
    length_n: State


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
parser.add_argument('--beam-size',
                    type=int,
                    default=5,
                    help='maximum beam size (default: 5)')
parser.add_argument('--max-lf-length',
                    type=int,
                    default=3,
                    help='maximum formula length (default: 3)')
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    help='batch size for computing segmentation masks (default: 128)')
parser.add_argument('--n-processes',
                    type=int,
                    default=8,
                    help='number of units to label in parallel (default: 8)')
parser.add_argument('--save-every',
                    type=int,
                    default=10,
                    help='save results after this many units finish')
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
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

# Cache these values early.
beam_size = args.beam_size
max_lf_length = args.max_lf_length
n_processes = args.n_processes
save_every = args.save_every

# Pick device.
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
if args.model == zoo.KEYS.ALEXNET and args.dataset == zoo.KEYS.PLACES365:
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
    print(f'----- labeling layer {layer} -----')

    layer = str(layer)
    cache_key = f'{args.model}_{args.dataset}_{layer}'

    # Begin by computing activation statistics.
    tally_cache_file = results_dir / f'{cache_key}_tally.npz'
    if generative:
        _, rq = dissect.generative(model,
                                   dataset,
                                   layer=layer,
                                   device=device,
                                   save_results=False,
                                   save_viz=False,
                                   tally_cache_file=tally_cache_file,
                                   **config.dissection.kwargs)
    else:
        _, rq = dissect.discriminative(model,
                                       dataset,
                                       layer=layer,
                                       device=device,
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
    # `segs`, int tensor
    #       of shape (n_images, n_labels_per_pixel, height, width)
    unit_masks = masks['unit_masks']
    seg_masks = masks['segs']

    # Let's precompute some expensive values that we will use over and over.
    # First, compute individual masks for each label.
    seg_literals_file = results_dir / f'{cache_key}_seg_literals.pth'
    if seg_literals_file.exists():
        print(f'loading seg literals from {seg_literals_file}')
        seg_literals = torch.load(seg_literals_file)
    else:
        seg_literals = [
            Literal(
                0,
                '-',
                torch.zeros(seg_masks.shape[0], *seg_masks.shape[2:]),
            )
        ]
        for index, label in tqdm(enumerate(seglabels),
                                 desc='precompute seg literals'):
            literal = Literal(index, label,
                              seg_masks.eq(index).sum().bool().long())
            seg_literals.append(literal)

        print(f'saving seg totals to {seg_literals_file}')
        torch.save(seg_literals, seg_literals_file)

    # Then compute the number of times each unit fired on any pixel.
    # (Again, useful for IOU.)
    unit_totals_file = results_dir / f'{cache_key}_unit_totals.pth'
    if unit_totals_file.exists():
        print(f'loading seg totals from {unit_totals_file}')
        unit_totals = torch.load(unit_totals_file)
    else:
        unit_totals = []
        for unit in tqdm(range(unit_masks.shape[1]),
                         desc='precompute unit totals'):
            unit_totals.append(unit_masks[:, unit].sum().item())

        print(f'saving unit totals to {unit_totals_file}')
        torch.save(unit_totals, unit_totals_file)

    # Finally, computing starting literals for each unit to limit search space.
    unit_literals_file = results_dir / f'{cache_key}_unit_literals.pth'
    if unit_literals_file.exists():
        print(f'loading unit literals from {unit_literals_file}')
        unit_literals = torch.load(unit_literals_file)
    else:
        unit_literals = []
        for unit in tqdm(range(unit_masks.shape[1]),
                         desc='precompute unit literals'):
            unique = seg_masks.mul(unit_masks[:, unit, None]).unique()
            unit_literals.append(
                [seg_literals[key] for key in unique.tolist() if key != 0])

        print(f'saving unit literals to {unit_literals_file}')
        torch.save(unit_literals, unit_literals_file)

    def beam_search(args: Tuple[int]) -> Result:
        """Run beam search for a single unit."""
        global seg_literals
        global unit_masks
        global unit_totals
        global unit_literals

        global beam_size
        global max_lf_length

        unit, = args
        beam = [
            State(literal, literal.iou(unit_masks[:, unit]))
            for literal in unit_literals[unit]
        ]
        beam = sorted(beam, key=lambda state: state.iou,
                      reverse=True)[:beam_size]
        best_length_1 = beam[0]

        for _ in range(max_lf_length - 1):
            frontier = []
            for state in beam:
                for literal in unit_literals[unit]:
                    for op, negate in ((Or, False), (And, False), (And, True)):
                        cand = op(state.logical_form,
                                  Not(literal) if negate else literal)
                        iou = cand.iou(unit_masks[:, unit])
                        frontier.append(State(cand, iou))
            beam = sorted(frontier, key=lambda state: state.iou,
                          reverse=True)[:beam_size]
        best_length_n = beam[0]

        return Result(unit, best_length_1, best_length_n)

    results_file = results_dir / f'{cache_key}_labels.csv'
    if results_file.exists():
        print(f'loading existing results from {results_file}')
        with results_file.open('r') as handle:
            rows = tuple(csv.DictReader(handle))
        completed = {int(row['unit']) for row in rows}
        units = tuple(set(range(unit_masks.shape[1])) - completed)
    else:
        units = tuple(range(unit_masks.shape[1]))

    header = ('layer', 'unit', 'max_length', 'label', 'iou')
    results = [header]
    with tqdm(total=unit_masks.shape[1], desc='label units') as progress:
        with multiprocessing.Pool(n_processes) as pool:
            inputs = [(unit,) for unit in units]
            for result in pool.imap_unordered(beam_search, inputs):
                progress.update()

                length_1 = result.length_1
                results.append((
                    layer,
                    str(result.unit),
                    '1',
                    str(length_1.logical_form),
                    f'{length_1.iou:.3f}',
                ))

                length_n = result.length_n
                results.append((
                    layer,
                    str(result.unit),
                    str(max_lf_length),
                    str(length_n.logical_form),
                    f'{length_n.iou:.3f}',
                ))

            finished = len(results) // 2
            if not finished % save_every:
                with results_file.open('w') as handle:
                    csv.writer(handle).writerows(results)
