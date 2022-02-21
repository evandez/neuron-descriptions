"""Train a CNN on spurious images with the class label in the corner."""
import argparse
import copy
import pathlib
import random
import shutil

from src import exemplars, milan, milannotations
from src.deps.netdissect import renormalize
from src.utils import ablations, env, training, viz
from src.utils.typing import StrSequence

import torch
import wandb
from torch import cuda
from tqdm.auto import tqdm

EXPERIMENTS = (
    exemplars.datasets.KEYS.IMAGENET_SPURIOUS_TEXT,
    exemplars.datasets.KEYS.IMAGENET_SPURIOUS_COLOR,
)

VERSION_ORIGINAL = 'original'
VERSION_5PCT = '5pct'
VERSION_10PCT = '10pct'
VERSION_50PCT = '50pct'
VERSION_100PCT = '100pct'
VERSIONS = (
    VERSION_ORIGINAL,
    VERSION_5PCT,
    VERSION_10PCT,
    VERSION_50PCT,
    VERSION_100PCT,
)

CONDITION_SORT_SPURIOUS = 'sort-spurious'
CONDITION_SORT_ALL = 'sort-all'
CONDITION_RANDOM = 'random'
CONDITIONS = (CONDITION_SORT_SPURIOUS, CONDITION_SORT_ALL, CONDITION_RANDOM)

parser = argparse.ArgumentParser(
    description='certify a cnn trained on bad data')
parser.add_argument('--experiments',
                    choices=EXPERIMENTS,
                    default=(exemplars.datasets.KEYS.IMAGENET_SPURIOUS_TEXT,),
                    nargs='+',
                    help='dataset to experiment with (default: all)')
parser.add_argument('--versions',
                    choices=VERSIONS,
                    default=(VERSION_50PCT,),
                    nargs='+',
                    help='versions of dataset to try (default: all)')
parser.add_argument('--conditions',
                    choices=CONDITIONS,
                    default=CONDITIONS,
                    nargs='+',
                    help='condition(s) to test under (default: all)')
parser.add_argument(
    '--cnn',
    choices=(exemplars.models.KEYS.ALEXNET, exemplars.models.KEYS.RESNET18),
    default=exemplars.models.KEYS.RESNET18,
    help='cnn architecture to train and certify (default: resnet18)')
parser.add_argument('--milan',
                    default='milan',
                    help='milan config (default: base)')
parser.add_argument(
    '--n-random-trials',
    type=int,
    default=5,
    help='for each experiment, delete an equal number of random '
    'neurons and retest this many times (default: 5)')
parser.add_argument('--fine-tune',
                    action='store_true',
                    help='fine tune last fully-connected cnn layers')
parser.add_argument('--no-mi',
                    action='store_true',
                    help='run the certification, but dont use MI decoding')
parser.add_argument('--milan-file',
                    type=pathlib.Path,
                    help='milan weights file (default: default pretrained)')
parser.add_argument('--data-dir',
                    type=pathlib.Path,
                    help='root dir for datasets (default: project data dir)')
parser.add_argument(
    '--results-dir',
    type=pathlib.Path,
    help='output directory to write models and dissection data '
    '(default: "<project results dir>/cnn-cert")')
parser.add_argument('--clear-results-dir',
                    action='store_true',
                    help='if set, clear results dir (default: do not)')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    help='training batch size (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='max training epochs (default: 100)')
parser.add_argument(
    '--patience',
    type=int,
    default=4,
    help='stop training if val loss worsens for this many epochs (default: 4)')
parser.add_argument(
    '--hold-out',
    type=float,
    default=.1,
    help='fraction of data to hold out for validation (default: .1)')
parser.add_argument('--lr',
                    type=float,
                    default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--ablation-min',
                    default=0,
                    help='min number of neurons to ablate (default: 0)')
parser.add_argument('--ablation-max',
                    type=int,
                    default=50,
                    help='max number of neurons to ablate (default: 50)')
parser.add_argument('--ablation-step-size',
                    default=1,
                    help='add\'l neurons to ablate at each step (default: 1)')
parser.add_argument('--device', help='manually set device (default: guessed)')
parser.add_argument('--wandb-project',
                    default='milan',
                    help='wandb project name (default: milan)')
parser.add_argument('--wandb-name', help='wandb run name (default: generated)')
parser.add_argument('--wandb-group',
                    default='experiments',
                    help='wandb group name (default: experiments)')
parser.add_argument('--wandb-n-samples',
                    type=int,
                    default=25,
                    help='number of samples to upload for each model')
args = parser.parse_args()

wandb.init(project=args.wandb_project,
           name=args.wandb_name or 'edit',
           group=args.wandb_group,
           config={
               'milan': args.milan,
               'cnn': args.cnn,
               'n_random_trials': args.n_random_trials,
               'fine_tune': bool(args.fine_tune),
           })

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

# Prepare necessary directories.
data_dir = args.data_dir or env.data_dir()

results_dir = args.results_dir
if results_dir is None:
    results_dir = env.results_dir() / 'edit'

if args.clear_results_dir and results_dir.exists():
    shutil.rmtree(results_dir)
results_dir.mkdir(exist_ok=True, parents=True)

# Load MILAN.
decoder = milan.pretrained(args.milan,
                           map_location=device,
                           path=args.milan_file)
encoder = decoder.encoder

# Now that we have MILAN, we can start the experiments.
for experiment in args.experiments:
    experiment_dir = results_dir / experiment
    experiment_dir.mkdir(exist_ok=True, parents=True)

    target_words: StrSequence
    if experiment == exemplars.datasets.KEYS.IMAGENET_SPURIOUS_TEXT:
        target_words = ('word', 'text', 'letter')
    else:
        assert experiment == exemplars.datasets.KEYS.IMAGENET_SPURIOUS_COLOR
        target_words = ('red', 'yellow', 'green', 'blue', 'cyan', 'purple',
                        'brown', 'black', 'white', 'gray')

    for version in args.versions:
        print(f'\n-------- BEGIN EXPERIMENT: {experiment}/{version} --------')

        # Start by training the classifier on spurious data.
        dataset = exemplars.datasets.load(
            experiment,
            factory=training.PreloadedImageFolder,
            path=data_dir / experiment / version / 'train',
        )
        test = exemplars.datasets.load(
            experiment,
            factory=training.PreloadedImageFolder,
            path=data_dir / experiment / version / 'test',
        )

        # Sample a small validation set for early stopping, etc.
        splits_file = experiment_dir / 'splits.pth'
        if splits_file.exists():
            print(f'reading train/val split from {splits_file}')
            splits = torch.load(splits_file)
            train, val = training.fixed_split(dataset, splits['val'])
        else:
            train, val = training.random_split(dataset, hold_out=args.hold_out)
            print(f'saving train/val splits to {splits_file}')
            torch.save({
                'train': train.indices,
                'val': val.indices
            }, splits_file)

        cnn, layers, _ = exemplars.models.load(
            f'{args.cnn}/{exemplars.datasets.KEYS.IMAGENET}',
            pretrained=False,
        )
        cnn = ablations.ImageClassifier(cnn).to(device)

        cnn_file = experiment_dir / f'{args.cnn}-{version}.pth'
        if cnn_file.exists():
            print(f'loading trained {args.cnn} from {cnn_file}')
            state_dict = torch.load(cnn_file, map_location=device)
            cnn.load_state_dict(state_dict)
        else:
            cnn.fit(dataset,
                    hold_out=val.indices,
                    batch_size=args.batch_size,
                    max_epochs=args.epochs,
                    patience=args.patience,
                    optimizer_kwargs={'lr': args.lr},
                    num_workers=0,
                    device=device,
                    display_progress_as=f'train {args.cnn}')
            print(f'saving trained {args.cnn} to {cnn_file}')
            torch.save(cnn.state_dict(), cnn_file)
        cnn.eval()

        # Now that we have the trained model, dissect it on the validation set.
        dissection_dir = experiment_dir / f'{args.cnn}-{version}'
        for layer in layers:
            print(f'dissecting: {layer}')
            exemplars.discriminative(
                cnn.model,
                val,
                layer=layer,
                results_dir=dissection_dir,
                tally_cache_file=dissection_dir / layer / 'tally.npz',
                masks_cache_file=dissection_dir / layer / 'masks.npz',
                device=device,
                # Have to manually set these since they cannot be inferred
                # from our custom dataset type.
                image_size=224,
                renormalizer=renormalize.renormalizer(source='imagenet',
                                                      target='byte'),
            )
        dissected = milannotations.TopImagesDataset(dissection_dir)

        descriptions_file = experiment_dir / f'{args.cnn}-{version}-descs.txt'
        if descriptions_file.exists():
            print(f'loading cached descriptions from {descriptions_file}')
            with descriptions_file.open('r') as handle:
                descriptions = handle.read().split('\n')
            assert len(descriptions) == len(dissected)
        else:
            descriptions = decoder.predict(
                dissected,
                strategy='beam' if args.no_mi else 'rerank',
                mi=False if args.no_mi else None,
                temperature=.2,
                beam_size=50,
                device=device)
            print(f'saving descriptions to {descriptions_file}')
            with descriptions_file.open('w') as handle:
                handle.write('\n'.join(descriptions))

        # Find candidate spurious neurons, and write them to disk.
        candidate_indices = [
            index for index, description in enumerate(descriptions)
            if any(word in description.lower() for word in target_words)
        ]
        candidates_file = experiment_dir / f'{args.cnn}-{version}-units.txt'
        print(f'found {len(candidate_indices)} candidate units; '
              f'saving to {candidates_file}')
        torch.save(candidate_indices, candidates_file)

        # Try cutting out each neuron individually, tracking its accuracy
        # on the validation dataset. This will help us filter out
        # important perceptual neurons.
        scores = None
        sort_spurious = CONDITION_SORT_SPURIOUS in args.conditions
        sort_all = CONDITION_SORT_ALL in args.conditions
        if sort_spurious or sort_all:
            scores_file = experiment_dir / f'{args.cnn}-{version}-scores.pth'
            if scores_file.exists():
                print(f'loading unit scores from {scores_file}')
                scores = torch.load(scores_file)
            else:
                scores = []
                for index in tqdm(range(len(dissected)), desc='score units'):
                    score = cnn.accuracy(val,
                                         ablate=[dissected.unit(index)],
                                         display_progress_as=None,
                                         num_workers=0,
                                         device=device)
                    scores.append(score)
                print(f'saving unit scores to {scores_file}')
                torch.save(scores, scores_file)

        # Compute its baseline accuracy on the test set.
        for condition in args.conditions:
            if condition == CONDITION_RANDOM:
                trials = args.n_random_trials
            else:
                trials = 1

            for trial in range(1, trials + 1):
                if condition == CONDITION_SORT_SPURIOUS:
                    assert scores is not None
                    indices = sorted(candidate_indices,
                                     key=scores.__getitem__,
                                     reverse=True)
                elif condition == CONDITION_SORT_ALL:
                    assert scores is not None
                    indices = sorted(range(len(dissected)),
                                     key=scores.__getitem__,
                                     reverse=True)[:len(candidate_indices)]
                else:
                    assert condition == CONDITION_RANDOM
                    indices = random.sample(range(len(dissected)),
                                            k=len(candidate_indices))

                ns_to_ablate = range(
                    args.ablation_min, args.ablation_max or
                    len(candidate_indices), args.ablation_step_size)
                for n_ablated in ns_to_ablate:
                    ablated_indices = indices[:n_ablated]
                    copied = copy.deepcopy(cnn)
                    if args.fine_tune:
                        copied.fit(
                            dataset,
                            hold_out=val.indices,
                            batch_size=args.batch_size,
                            max_epochs=args.epochs,
                            patience=args.patience,
                            optimizer_kwargs={'lr': args.lr},
                            ablate=dissected.units(ablated_indices),
                            layers=['fc']
                            if args.cnn == exemplars.models.KEYS.RESNET18 else
                            ['fc6', 'fc7', 'linear8'],
                            num_workers=0,
                            device=device,
                            display_progress_as=f'fine tune {args.cnn} '
                            f'(cond={condition}, t={trial}, n={n_ablated})')
                    accuracies = {}
                    for key, evaluation in (('val', val), ('test', test)):
                        accuracies[key] = copied.accuracy(
                            evaluation,
                            ablate=dissected.units(ablated_indices),
                            display_progress_as=f'compute {key} accuracy '
                            f'(cond={condition}, t={trial}, n={n_ablated})',
                            num_workers=0,
                            device=device,
                        )
                    samples = viz.random_neuron_wandb_images(
                        dissected,
                        descriptions,
                        indices=ablated_indices,
                        k=args.wandb_n_samples,
                        exp=experiment,
                        ver=version,
                        cond=condition,
                        n_ablated=n_ablated)
                    wandb.log({
                        'experiment': experiment,
                        'version': version,
                        'condition': condition,
                        'trial': trial,
                        'n_ablated': n_ablated,
                        'samples': samples,
                        **accuracies,
                    })
                    print(f'experiment={experiment}', f'version={version}',
                          f'condition={condition}', f'trial={trial}',
                          f'n_ablated={n_ablated}', f'val={accuracies["val"]}',
                          f'test={accuracies["test"]}')
