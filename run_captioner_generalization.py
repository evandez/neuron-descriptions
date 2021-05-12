"""Run captioner generalization experiments."""
import argparse
import pathlib
from typing import Sized, cast

from lv import zoo
from lv.models import annotators, captioners, featurizers

import wandb
from torch.utils import data

EXPERIMENTS = {
    'within-network': ((
        'alexnet/imagenet',
        'alexnet/places365',
        'resnet152/imagenet',
        'resnet152/places365',
        'biggan/imagenet'
        'biggan/places365',
    ),),
    'across-network': (
        ('alexnet/imagenet', 'alexnet/places365'),
        ('resnet152/imagenet', 'resnet152/places365'),
    ),
    'across-dataset': (
        ('alexnet/imagenet', 'resnet152/imagenet', 'biggan/imagenet'),
        ('alexnet/places365', 'resnet152/places365', 'biggan/places365'),
    ),
    'across-task': (
        (
            'alexnet/imagenet',
            'alexnet/places365',
            'resnet152/imagenet',
            'resnet152/places365',
        ),
        ('biggan/imagenet', 'biggan/places365'),
    ),
}

parser = argparse.ArgumentParser(description='run captioner abalations')
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
                    default='experiments',
                    help='wandb group name')
parser.add_argument('--wandb-entity', help='wandb user or team')
parser.add_argument('--wandb-dir', metavar='PATH', help='wandb directory')
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

for experiment in args.experiments or EXPERIMENTS.keys():
    splits = EXPERIMENTS[experiment]
    if len(splits) == 2:
        left = zoo.datasets(*splits[0], path=args.dataset_root)
        right = zoo.datasets(*splits[1], path=args.dataset_root)
        configs = ((left, right, *splits), (right, left, *reversed(splits)))
    else:
        assert len(splits) == 1, 'weird splits?'
        names, = splits
        configs = []
        for name in names:
            dataset = zoo.datasets(name, path=args.dataset_root)
            size = len(cast(Sized, dataset))
            test_size = (.1 * size)
            train_size = size - test_size
            train, test = data.random_split(dataset, (train_size, test_size))
            configs.append((train, test, (name,), (name,)))

    for train, test, train_keys, test_keys in configs:
        featurizer = featurizers.MaskedPyramidFeaturizer().to(device)
        train_features = featurizer.map(train, device=device)
        test_features = featurizer.map(test, device=device)

        annotator = annotators.WordAnnotator.fit(
            train,
            featurizer,
            indexer_kwargs={'ignore_rarer_than': 5},
            features=train_features)

        captioner = captioners.decoder(train,
                                       featurizer=featurizer,
                                       annotator=annotator)
        captioner.fit(train, features=train_features, device=device)
        predictions = captioner.predict(test,
                                        features=test_features,
                                        device=device)
        bleu = captioner.bleu(test, predictions=predictions)
        rouge = captioner.rouge(test, predictions=predictions)

        log = {
            'condition': experiment,
            'train': train_keys,
            'test': test_keys,
            'bleu': bleu.score,
        }
        for index, precision in enumerate(bleu.precisions):
            log[f'bleu-{index + 1}'] = precision
        for kind, scores in rouge.items():
            for key, score in scores.items():
                log[f'{kind}-{key}'] = score
        log['samples'] = [
            wandb.Image(
                test[index].as_pil_image_grid(),
                caption=predictions[index],
            ) for index in test.indices[:25]
        ]
        wandb.log(log)
