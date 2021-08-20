"""Run captioner generalization experiments."""
import argparse
import pathlib
from typing import Mapping, Sequence, Sized, Tuple, cast

from lv import zoo
from lv.ext import bert_score
from lv.models import decoders, encoders

import wandb
from torch.utils import data

DatasetNames = Sequence[str]
Splits = Tuple[DatasetNames, ...]

EXPERIMENT_WITHIN_NETWORK = 'within-network'
EXPERIMENT_ACROSS_NETWORK = 'across-network'
EXPERIMENT_ACROSS_DATASET = 'across-dataset'
EXPERIMENT_ACROSS_TASK = 'across-task'
EXPERIMENT_LEAVE_ONE_OUT = 'leave-one-out'
EXPERIMENTS: Mapping[str, Splits] = {
    EXPERIMENT_WITHIN_NETWORK: ((
        'alexnet/imagenet',
        'alexnet/places365',
        'resnet152/imagenet',
        'resnet152/places365',
        'biggan/imagenet',
        'biggan/places365',
    ),),
    EXPERIMENT_ACROSS_NETWORK: (
        ('alexnet/imagenet', 'alexnet/places365'),
        ('resnet152/imagenet', 'resnet152/places365'),
    ),
    EXPERIMENT_ACROSS_DATASET: (
        (
            'alexnet/imagenet',
            'resnet152/imagenet',
            'biggan/imagenet',
        ),
        (
            'alexnet/places365',
            'resnet152/places365',
            'biggan/places365',
        ),
    ),
    EXPERIMENT_ACROSS_TASK: (
        (
            'alexnet/imagenet',
            'alexnet/places365',
            'resnet152/imagenet',
            'resnet152/places365',
        ),
        ('biggan/imagenet', 'biggan/places365'),
    ),
    EXPERIMENT_LEAVE_ONE_OUT: ((
        'alexnet/imagenet',
        'alexnet/places365',
        'resnet152/imagenet',
        'resnet152/places365',
        'biggan/imagenet',
        'biggan/places365',
    ),)
}

parser = argparse.ArgumentParser(
    description='run captioner generalization experiments')
parser.add_argument('--experiments',
                    nargs='+',
                    help='experiments to run (default: all experiments)')
parser.add_argument('--datasets-root',
                    type=pathlib.Path,
                    help='root dir for datasets (default: .zoo/datasets)')
parser.add_argument('--hold-out',
                    type=float,
                    default=.1,
                    help='hold out this fraction of data for testing')
parser.add_argument('--wandb-project',
                    default='lv',
                    help='wandb project name (default: lv)')
parser.add_argument('--wandb-name',
                    default='captioner-generalization',
                    help='wandb run name (default: captioner-generalization)')
parser.add_argument('--wandb-group',
                    default='captioner',
                    help='wandb group name (default: captioner)')
parser.add_argument('--wandb-entity', help='wandb user or team')
parser.add_argument('--wandb-dir', metavar='PATH', help='wandb directory')
parser.add_argument('--wandb-n-samples',
                    type=int,
                    default=25,
                    help='number of samples to upload for each model')
parser.add_argument('--cuda', action='store_true', help='use cuda device')
args = parser.parse_args()

wandb.init(project=args.wandb_project,
           name=args.wandb_name,
           entity=args.wandb_entity,
           group=args.wandb_group,
           dir=args.wandb_dir)
run = wandb.run
assert run is not None, 'failed to initialize wandb?'

device = 'cuda' if args.cuda else 'cpu'

# Load BERTScorer once up front.
bert_scorer = bert_score.BERTScorer(lang='en',
                                    rescale_with_baseline=True,
                                    device=device)

# Load encoder.
encoder = encoders.PyramidConvEncoder().to(device)

# Start experiments.
for experiment in args.experiments or EXPERIMENTS.keys():
    print(f'\n-------- BEGIN EXPERIMENT: {experiment} --------')

    # Have to handle within-network and across-* experiments differently.
    splits = EXPERIMENTS[experiment]
    if len(splits) == 2:
        left = zoo.datasets(*splits[0], path=args.datasets_root)
        right = zoo.datasets(*splits[1], path=args.datasets_root)
        configs = [(left, right, *splits), (right, left, *reversed(splits))]
    elif experiment == EXPERIMENT_WITHIN_NETWORK:
        assert len(splits) == 1
        names, = splits
        configs = []
        for name in names:
            dataset = zoo.datasets(name, path=args.datasets_root)
            size = len(cast(Sized, dataset))
            test_size = int(.1 * size)
            train_size = size - test_size
            split = data.random_split(dataset, (train_size, test_size))
            configs.append((*split, (name,), (name,)))
    else:
        assert len(splits) == 1
        assert experiment == EXPERIMENT_LEAVE_ONE_OUT

        names, = splits

        datasets_by_name = {}
        for name in names:
            dataset = zoo.datasets(name, path=args.datasets_root)
            datasets_by_name[name] = dataset

        unique = set(names)
        configs = []
        for name in unique:

            test_keys = (name,)
            test = datasets_by_name[name]

            train_keys = unique - {name}
            assert train_keys

            train = None
            for other in train_keys:
                if train is None:
                    train = datasets_by_name[other]
                else:
                    train += datasets_by_name[other]

            configs.append((train, test, train_keys, test_keys))

    # For every train/test set, train the captioner, test it, and log.
    for train, test, train_keys, test_keys in configs:
        train_features = encoder.map(cast(data.Dataset, train), device=device)
        test_features = encoder.map(cast(data.Dataset, test), device=device)

        # Train the model.
        decoder = decoders.decoder(train, encoder)
        decoder.fit(train, features=train_features, device=device)
        predictions = decoder.predict(test,
                                      features=test_features,
                                      device=device)
        bleu = decoder.bleu(test, predictions=predictions)
        rouge = decoder.rouge(test, predictions=predictions)
        bert_scores = decoder.bert_score(test,
                                         predictions=predictions,
                                         bert_scorer=bert_scorer)

        # Log ALL the things!
        log = {
            'experiment': experiment,
            'train': tuple(train_keys),
            'test': tuple(test_keys),
            'bleu': bleu.score,
        }
        for index, precision in enumerate(bleu.precisions):
            log[f'bleu-{index + 1}'] = precision
        for kind, scores in rouge.items():
            for key, score in scores.items():
                log[f'{kind}-{key}'] = score
        for kind, score in bert_scores.items():
            log[f'bert_score-{kind}'] = score
        log['samples'] = [
            wandb.Image(
                test[index].as_pil_image_grid(),
                caption=f'({experiment.upper()}) {predictions[index]}',
            ) for index in range(min(len(test), args.wandb_n_samples))
        ]
        wandb.log(log)
