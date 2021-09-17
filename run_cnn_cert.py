"""Train a CNN on spurious images with the class label in the corner."""
import argparse
import copy
import pathlib
import random
import shutil

import lv.zoo
from lv import datasets, models
from lv.deps.netdissect import renormalize
from lv.dissection import dissect, zoo
from lv.utils import env, training, viz

import numpy
import torch
import wandb
from torch import cuda

EXPERIMENTS = (zoo.KEY_SPURIOUS_IMAGENET_TEXT,)

VERSIONS = (
    # 'original',
    '5pct',
    '10pct',
    '25pct',
    # '50pct',
    # '100pct',
)

CONDITION_TEXT = 'ablate-text'
CONDITION_RANDOM = 'ablate-random'
CONDITIONS = (CONDITION_TEXT, CONDITION_RANDOM)

parser = argparse.ArgumentParser(
    description='certify a cnn trained on bad data')
parser.add_argument('--experiments',
                    choices=EXPERIMENTS,
                    default=EXPERIMENTS,
                    nargs='+',
                    help='dataset to experiment with (default: all)')
parser.add_argument('--versions',
                    choices=VERSIONS,
                    default=VERSIONS,
                    nargs='+',
                    help='versions of dataset to try (default: all)')
parser.add_argument('--conditions',
                    choices=CONDITIONS,
                    default=CONDITIONS,
                    nargs='+',
                    help='condition(s) to test under (default: all)')
parser.add_argument(
    '--cnn',
    choices=(lv.zoo.KEY_ALEXNET, zoo.KEY_RESNET18),
    default=zoo.KEY_RESNET18,
    help='cnn architecture to train and certify (default: resnet18)')
parser.add_argument('--captioner',
                    nargs=2,
                    default=(lv.zoo.KEY_CAPTIONER_RESNET101, lv.zoo.KEY_ALL),
                    help='captioner model (default: captioner-resnet101 all)')
parser.add_argument(
    '--n-random-trials',
    type=int,
    default=5,
    help='for each experiment, delete an equal number of random '
    'neurons and retest this many times (default: 5)')
parser.add_argument('--fine-tune',
                    action='store_true',
                    help='fine tune last fully-connected cnn layers')
parser.add_argument('--captioner-file',
                    type=pathlib.Path,
                    help='captioner weights file (default: loaded from zoo)')
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
                    type=float,
                    default=0,
                    help='min fraction of neurons to ablate (default: 0)')
parser.add_argument('--ablation-max',
                    type=float,
                    default=1,
                    help='max fraction of neurons to ablate (default: 1)')
parser.add_argument(
    '--ablation-step-size',
    type=float,
    default=.1,
    help='fraction of additional neurons to ablate at each step (default: .1)')
parser.add_argument('--device', help='manually set device (default: guessed)')
parser.add_argument('--wandb-project',
                    default='lv',
                    help='wandb project name (default: lv)')
parser.add_argument('--wandb-name',
                    default='cnn-cert',
                    help='wandb run name (default: cnn-cert)')
parser.add_argument('--wandb-group',
                    default='applications',
                    help='wandb group name (default: applications)')
parser.add_argument('--wandb-n-samples',
                    type=int,
                    default=25,
                    help='number of samples to upload for each model')
args = parser.parse_args()

wandb.init(project=args.wandb_project,
           name=args.wandb_name,
           group=args.wandb_group,
           config={
               'captioner': '/'.join(args.captioner),
               'cnn': args.cnn,
               'n_random_trials': args.n_random_trials,
               'fine_tune': bool(args.fine_tune),
           })

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

# Prepare necessary directories.
data_dir = args.data_dir or env.data_dir()

results_dir = args.results_dir
if results_dir is None:
    results_subdir = f'cnn-cert-r{args.n_random_trials}'
    if args.fine_tune:
        results_subdir += '-ft'
    results_dir = env.results_dir() / results_subdir

if args.clear_results_dir and results_dir.exists():
    shutil.rmtree(results_dir)
results_dir.mkdir(exist_ok=True, parents=True)

# Load the captioner.
captioner_model, captioner_dataset = args.captioner
decoder, _ = lv.zoo.model(captioner_model,
                          captioner_dataset,
                          path=args.captioner_file,
                          map_location=device)
encoder = decoder.encoder
assert isinstance(decoder, models.Decoder)
assert isinstance(encoder, models.Encoder)

# Now that we have the captioner, we can start the experiments.
for experiment in args.experiments:
    for version in args.versions:
        print(f'\n-------- BEGIN EXPERIMENT: {experiment}/{version} --------')

        # Start by training the classifier on spurious data.
        dataset = zoo.dataset(experiment,
                              path=data_dir / experiment / version / 'train',
                              factory=training.PreloadedImageFolder)
        test = zoo.dataset(experiment,
                           path=data_dir / experiment / version / 'test',
                           factory=training.PreloadedImageFolder)
        train, val = training.random_split(dataset, hold_out=args.hold_out)

        cnn, layers, _ = zoo.model(args.cnn,
                                   zoo.KEY_IMAGENET,
                                   pretrained=False)
        cnn = models.classifier(cnn).to(device)
        cnn.fit(train,
                hold_out=val.indices,
                batch_size=args.batch_size,
                max_epochs=args.epochs,
                patience=args.patience,
                optimizer_kwargs={'lr': args.lr},
                device=device,
                display_progress_as=f'train {args.cnn}')
        torch.save(cnn.state_dict(),
                   results_dir / experiment / f'{args.cnn}-{version}.pth')

        # Now that we have the trained model, dissect it on the validation set.
        dissection_root = results_dir / experiment / version / args.cnn
        for layer in layers:
            dissect.discriminative(
                cnn,
                val,
                layer=layer,
                results_dir=dissection_root,
                clear_results_dir=True,
                device=device,
                # TODO(evandez): Remove need for these arguments...
                image_size=224,
                renormalizer=renormalize.renormalizer(source='imagenet',
                                                      target='byte'),
            )
        dissected = datasets.TopImagesDataset(dissection_root)
        captions = decoder.predict(dissected, device=device)
        texts = [
            index for index, caption in enumerate(captions)
            if 'text' in caption
        ]

        # Compute its baseline accuracy on the test set.
        for condition in args.conditions:
            if condition == CONDITION_RANDOM:
                trials = args.n_random_trials
            else:
                trials = 1

            for trial in range(trials):
                if condition == CONDITION_TEXT:
                    indices = texts
                else:
                    assert condition == CONDITION_RANDOM
                    indices = random.sample(range(len(dissected)),
                                            k=len(texts))

                fractions = numpy.arange(args.ablation_min, args.ablation_max,
                                         args.ablation_step_size)
                for fraction in fractions:
                    ablated = indices[:int(fraction * len(indices))]
                    copied = copy.deepcopy(cnn)
                    if args.fine_tune:
                        copied.fit(
                            train,
                            hold_out=val.indices,
                            batch_size=args.batch_size,
                            max_epochs=args.epochs,
                            patience=args.patience,
                            optimizer_kwargs={'lr': args.lr},
                            ablate=dissected.units(ablated),
                            layers=['fc'] if args.cnn == zoo.KEY_RESNET18 else
                            ['fc6', 'fc7', 'linear8'],
                            device=device,
                            display_progress_as=f'fine tune {args.cnn}')
                    accuracy = copied.accuracy(
                        test,
                        ablate=dissected.units(ablated),
                        display_progress_as=f'test ablated {args.cnn} '
                        f'(exp={experiment}, '
                        f'ver={version}, '
                        f'cond={condition}, '
                        f'trial={trial + 1}, '
                        f'frac={fraction})',
                        device=device,
                    )
                    samples = viz.random_neuron_wandb_images(
                        dissected,
                        captions,
                        indices=ablated,
                        k=args.wandb_n_samples,
                        exp=experiment,
                        ver=version,
                        cond=condition,
                        frac=fraction)
                    wandb.log({
                        'experiment': experiment,
                        'version': version,
                        'condition': condition,
                        'trial': trial,
                        'frac_ablated': fraction,
                        'n_ablated': len(ablated),
                        'accuracy': accuracy,
                        'samples': samples,
                    })
