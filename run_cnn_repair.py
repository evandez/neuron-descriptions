"""Train a CNN on spurious images with the class label in the corner."""
import argparse
import copy
import pathlib
import random

import lv.zoo
from lv import datasets
from lv.dissection import dissect, zoo
from lv.models import annotators, captioners, classifiers, featurizers
from lv.utils import logging, training
from third_party.netdissect import renormalize

import numpy as np
import wandb

EXPERIMENTS = (
    zoo.KEY_SPURIOUS_IMAGENET,
    # TODO(evandez): Figure out what to do with this one.
    # zoo.KEY_SPURIOUS_PLACES365,
)
VERSIONS = ('original', 'spurious-5', 'spurious-10', 'spurious-25',
            'spurious-100')

CONDITION_TEXT = 'ablate-text'
CONDITION_RANDOM = 'ablate-random'
CONDITIONS = (CONDITION_TEXT, CONDITION_RANDOM)

ANNOTATIONS = (
    lv.zoo.KEY_ALEXNET_IMAGENET,
    lv.zoo.KEY_ALEXNET_PLACES365,
    lv.zoo.KEY_RESNET152_IMAGENET,
    lv.zoo.KEY_RESNET152_PLACES365,
    lv.zoo.KEY_BIGGAN_IMAGENET,
    lv.zoo.KEY_BIGGAN_PLACES365,
)

CAPTIONER_SAT = 'sat'
CAPTIONER_SAT_MF = 'sat+mf'
CAPTIONER_SAT_WF = 'sat+wf'
CAPTIONER_SAT_MF_WF = 'sat+mf+wf'
CAPTIONERS = (CAPTIONER_SAT, CAPTIONER_SAT_MF, CAPTIONER_SAT_WF,
              CAPTIONER_SAT_MF_WF)

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
parser.add_argument('--conditions',
                    choices=CONDITIONS,
                    default=CONDITIONS,
                    nargs='+',
                    help='condition(s) to test under (default: all)')
parser.add_argument('--captioner',
                    choices=CAPTIONERS,
                    default=CAPTIONER_SAT_MF,
                    help='captioning model to use (default: sat+mf)')
parser.add_argument('--cnn',
                    choices=(lv.zoo.KEY_ALEXNET, zoo.KEY_RESNET18),
                    default=zoo.KEY_RESNET18,
                    help='cnn architecture to repair')
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
           config={
               'captioner': args.captioner,
               'n_random_trials': args.n_random_trials,
           },
           dir=args.wandb_dir)

device = 'cuda' if args.cuda else 'cpu'

args.out_root.mkdir(exist_ok=True, parents=True)

# Before diving into experiments, train a captioner on all the data.
# TODO(evandez): Use a pretrained captioner.
annotations = lv.zoo.datasets(*args.annotations, path=args.datasets_root)

if args.captioner != CAPTIONER_SAT:
    featurizer = featurizers.MaskedPyramidFeaturizer().to(device)
else:
    featurizer = featurizers.MaskedImagePyramidFeaturizer().to(device)

features = None
if args.captioner != CAPTIONER_SAT_WF:
    features = featurizer.map(annotations, device=device)

annotator = None
if args.captioner in (CAPTIONER_SAT_WF, CAPTIONER_SAT_MF_WF):
    annotator = annotators.WordAnnotator.fit(annotations,
                                             featurizer=featurizer,
                                             features=features,
                                             device=device)

if args.captioner in (CAPTIONER_SAT, CAPTIONER_SAT_MF):
    captioner = captioners.decoder(annotations, featurizer=featurizer)
else:
    captioner = captioners.decoder(
        annotations,
        annotator=annotator,
        featurizer=None if args.captioner == CAPTIONER_SAT_WF else featurizer)

captioner.fit(annotations,
              features=features,
              display_progress_as=f'train {args.captioner}',
              device=device)

# Now that we have the captioner, we can start the experiments.
for experiment in args.experiments:
    for version in args.versions:
        print(f'\n-------- BEGIN EXPERIMENT: {experiment}/{version} --------')

        # Start by training the classifier on spurious data.
        dataset = zoo.dataset(experiment,
                              path=args.datasets_root / experiment / version /
                              'train',
                              factory=training.PreloadedImageFolder)
        test = zoo.dataset(experiment,
                           path=args.datasets_root / experiment / version /
                           'test',
                           factory=training.PreloadedImageFolder)
        train, val = training.random_split(dataset, hold_out=args.hold_out)

        cnn, layers, _ = zoo.model(args.cnn,
                                   zoo.KEY_IMAGENET,
                                   pretrained=False)
        cnn = classifiers.ImageClassifier(cnn).to(device)
        cnn.fit(train,
                hold_out=val,
                batch_size=args.batch_size,
                max_epochs=args.epochs,
                patience=args.patience,
                optimizer_kwargs={'lr': args.lr},
                device=device,
                display_progress_as=f'train {args.cnn}')

        # Now that we have the trained model, dissect it on the validation set.
        dissection_root = args.out_root / experiment / version / args.cnn
        for layer in layers:
            dissect.sequential(
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
        captions = captioner.predict(dissected, device=device)
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

                fractions = np.arange(args.ablation_min, args.ablation_max,
                                      args.ablation_step_size)
                for fraction in fractions:
                    ablated = indices[:int(fraction * len(indices))]
                    copied = copy.deepcopy(cnn)
                    copied.fit(train,
                               hold_out=val,
                               batch_size=args.batch_size,
                               max_epochs=args.epochs,
                               patience=args.patience,
                               optimizer_kwargs={'lr': args.lr},
                               ablate=dissected.units(ablated),
                               layers=['fc'] if args.cnn == zoo.KEY_RESNET18
                               else ['fc6', 'fc7', 'linear8'],
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
                    samples = logging.random_neuron_wandb_images(
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
