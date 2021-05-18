"""Train a CNN on spurious images with the class label in the corner."""
import argparse
import pathlib
import random
from typing import Sized, cast

import lv.zoo
import run_cnn_ablations
from lv import datasets
from lv.dissection import dissect, zoo
from lv.models import captioners, featurizers
from lv.utils import training

import torch
import wandb
from torch import nn, optim
from torch.utils import data
from tqdm.auto import tqdm

EXPERIMENTS = (zoo.KEY_SPURIOUS_IMAGENET, zoo.KEY_SPURIOUS_PLACES365)
VERSIONS = ('original', 'spurious')

ANNOTATIONS = (
    lv.zoo.KEY_ALEXNET_IMAGENET,
    # TODO(evandez): Uncomment once ready.
    # lv.zoo.KEY_ALEXNET_PLACES365,
    # lv.zoo.KEY_RESNET152_IMAGENET,
    # lv.zoo.KEY_RESNET152_PLACES365,
    # lv.zoo.KEY_BIGGAN_IMAGENET,
    # lv.zoo.KEY_BIGGAN_PLACES365,
)

parser = argparse.ArgumentParser(description='train a cnn on spurious data')
parser.add_argument('--experiments',
                    choices=EXPERIMENTS,
                    default=EXPERIMENTS,
                    nargs='+',
                    help='train one model for each of these datasets')
parser.add_argument('--versions',
                    choices=VERSIONS,
                    default=VERSIONS,
                    nargs='+',
                    help='version(s) of each dataset to use')
parser.add_argument(
    '--n-random-trials',
    type=int,
    default=5,
    help='for each experiment, delete an equal number of random '
    'neurons and retest this many times (default: 5)')
parser.add_argument('--annotations',
                    choices=ANNOTATIONS,
                    default=ANNOTATIONS,
                    nargs='+',
                    help='annotations to train captioner on (default: all)')
parser.add_argument('--datasets-root',
                    type=pathlib.Path,
                    default='.zoo/datasets',
                    help='root dir for datasets (default: .zoo/datasets)')
parser.add_argument(
    '--out-root',
    type=pathlib.Path,
    default='train-spurious-cnn',
    help='output directory to write models and dissection data (default: ".")')
parser.add_argument(
    '--num-workers',
    type=int,
    default=32,
    help='number of worker threads to load data with (default: 32)')
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
parser.add_argument('--cuda', action='store_true', help='use cuda device')
parser.add_argument('--wandb-project',
                    default='lv',
                    help='wandb project name (default: lv)')
parser.add_argument('--wandb-name',
                    default='cnn-repair',
                    help='wandb run name (default: cnn-repair)')
parser.add_argument('--wandb-group',
                    default='applications',
                    help='wandb group name (default: applications)')
parser.add_argument('--wandb-entity', help='wandb user or team')
parser.add_argument('--wandb-dir', metavar='PATH', help='wandb directory')
parser.add_argument('--wandb-n-samples',
                    type=int,
                    default=25,
                    help='number of samples to upload for each model')
args = parser.parse_args()

wandb.init(project=args.wandb_project,
           name=args.wandb_name,
           entity=args.wandb_entity,
           group=args.wandb_group,
           dir=args.wandb_dir)

device = 'cuda' if args.cuda else 'cpu'

args.out_root.mkdir(exist_ok=True, parents=True)

# Before diving into experiments, train a captioner on all the data.
# TODO(evandez): Use a pretrained captioner.
annotations = lv.zoo.datasets(*args.annotations, path=args.datasets_root)
featurizer = featurizers.MaskedPyramidFeaturizer().to(device)
captioner = captioners.decoder(annotations, featurizer=featurizer)
captioner.fit(annotations, device=device)

for experiment in args.experiments:
    for version in args.versions:
        print(f'\n-------- BEGIN EXPERIMENT: {experiment}/{version} --------')

        # Start by training the classifier on spurious data.
        dataset = zoo.dataset(experiment,
                              path=args.datasets_root / experiment / version /
                              'train')
        test = zoo.dataset(experiment,
                           path=args.datasets_root / experiment / version /
                           'test')
        size = len(cast(Sized, dataset))
        val_size = int(args.hold_out * size)
        train_size = size - val_size
        train, val = data.random_split(dataset, (train_size, val_size))
        train_loader = data.DataLoader(train,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       batch_size=args.batch_size)
        val_loader = data.DataLoader(val,
                                     num_workers=args.num_workers,
                                     batch_size=args.batch_size)

        model, layers, _ = zoo.model(zoo.KEY_RESNET18,
                                     zoo.KEY_IMAGENET,
                                     pretrained=False)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        stopper = training.EarlyStopping(patience=args.patience)

        desc = f'train {zoo.KEY_RESNET18}'
        progress = tqdm(range(args.epochs), desc=desc)
        for _ in progress:
            model.train()
            train_loss = 0.
            for images, targets in train_loader:
                predictions = model(images.to(device))
                loss = criterion(predictions, targets.to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.
            for images, targets in val_loader:
                with torch.no_grad():
                    predictions = model(images.to(device))
                    val_loss += criterion(predictions,
                                          targets.to(device)).item()
            val_loss /= len(val_loader)

            progress.set_description(f'{desc} [train_loss={train_loss:.3f}, '
                                     f'val_loss={val_loss:.3f}]')

            if stopper(val_loss):
                break

        # Now that we have the trained model, dissect it on the validation set.
        dissection_root = args.out_root / f'{zoo.KEY_RESNET18}-{experiment}'
        for layer in tqdm(layers, desc='dissect resnet18'):
            dissect.sequential(model,
                               val,
                               layer=layer,
                               results_dir=dissection_root,
                               num_workers=args.num_workers,
                               device=device)

        # Now find spurious neurons and cut them out.
        dissected = datasets.TopImagesDataset(dissection_root)
        captions = captioner.predict(dissected, device=device)
        indices = [
            index for index, caption in enumerate(captions)
            if 'text' in caption
        ]
        accuracy = run_cnn_ablations.ablate_and_test(
            model,
            test,
            dissected,
            indices,
            display_progress_as=f'ablate text neurons (n={len(indices)})')
        samples = run_cnn_ablations.create_wandb_images(
            dissected,
            captions,
            indices,
            condition=f'{experiment}/{version}/text',
            k=args.wandb_n_samples)
        wandb.log({
            'experiment': experiment,
            'version': version,
            'condition': 'text',
            'trial': 1,
            'neurons': len(indices),
            'accuracy': accuracy,
            'samples': samples,
        })

        # Repeat the experiment on random neurons so we have a baseline.
        for trial in range(args.n_random_trials):
            indices = random.sample(range(len(dissected)), k=len(indices))
            accuracy = run_cnn_ablations.ablate_and_test(
                model,
                test,
                dissected,
                indices,
                display_progress_as=f'ablate random '
                f'(trial={trial + 1}, n={len(indices)})')
            samples = run_cnn_ablations.create_wandb_images(
                dissected,
                captions,
                indices,
                condition=f'{experiment}/{version}/random',
                k=args.wandb_n_samples)
            wandb.log({
                'experiment': experiment,
                'version': version,
                'condition': 'random',
                'trial': trial + 1,
                'neurons': len(indices),
                'accuracy': accuracy,
                'samples': samples,
            })
