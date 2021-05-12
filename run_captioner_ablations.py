"""Script for running all captioner ablations."""
import argparse
import pathlib
from typing import Sized, cast

from lv import zoo
from lv.models import annotators, captioners, featurizers

from torch.utils import data
import wandb

parser = argparse.ArgumentParser(description='run captioner abalations')
parser.add_argument('--datasets',
                    default=('alexnet/imagenet', 'alexnet/places365'),
                    nargs='+',
                    help='training datasets (default: all datasets)')
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
                    default='captioner-ablations',
                    help='wandb run name (default: captioner-ablations)')
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

dataset = zoo.datasets(*args.datasets)

size = len(cast(Sized, dataset))
test_size = int(args.hold_out * size)
train_size = size - test_size
train, test = data.random_split(dataset, (train_size, test_size))

featurizer = featurizers.MaskedPyramidFeaturizer().to(device)
train_features = featurizer.map(train, device=device)
test_features = featurizer.map(test, device=device)

annotator = annotators.WordAnnotator.fit(
    train,
    featurizer,
    indexer_kwargs={'ignore_rarer_than': 5},
    features=train_features)

ablations = {
    'sat':
        captioners.decoder(
            train, featurizer=featurizers.MaskedImagePyramidFeaturizer()),
    'sat+mf':
        captioners.decoder(train, featurizer=featurizer),
    'sat+wf':
        captioners.decoder(train, annotator=annotator),
    'sat+mf+wf':
        captioners.decoder(train, featurizer=featurizer, annotator=annotator),
}
for condition, captioner in ablations.items():
    captioner.fit(train,
                  features=train_features if condition != 'sat' else None,
                  device=device)
    predictions = captioner.predict(
        test,
        features=test_features if condition != 'sat' else None,
        device=device)
    bleu = captioner.bleu(test, predictions=predictions)
    rouge = captioner.rouge(test, predictions=predictions)

    log = {'condition': condition, 'bleu': bleu.score}
    for index, precision in enumerate(bleu.precisions):
        log[f'bleu-{index + 1}'] = precision
    for kind, scores in rouge.items():
        for key, score in scores.items():
            log[f'{kind}-{key}'] = score
    log['samples'] = [
        wandb.Image(
            test[index].as_pil_image_grid(),
            caption=f'({condition.upper()}) {predictions[index]}',
        ) for index in test.indices[:25]
    ]
    wandb.log(log)
