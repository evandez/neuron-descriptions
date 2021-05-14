"""Run captioner generalization experiments."""
import argparse
import pathlib
from typing import Mapping, Optional, Sequence, Sized, Tuple, cast

from lv import zoo
from lv.models import annotators, captioners, featurizers

import wandb
from torch.utils import data

DatasetNames = Sequence[str]
Splits = Tuple[DatasetNames, ...]

EXPERIMENTS: Mapping[str, Splits] = {
    'within-network': (
        (
            'alexnet/imagenet',
            'alexnet/places365',
            'resnet152/imagenet',
            'resnet152/places365',
            # TODO(evandez): Uncomment when data is available.
            # 'biggan/imagenet'
            # 'biggan/places365',
        ),),
    'across-network': (
        ('alexnet/imagenet', 'alexnet/places365'),
        ('resnet152/imagenet', 'resnet152/places365'),
    ),
    'across-dataset': (
        (
            'alexnet/imagenet',
            'resnet152/imagenet',
            # TODO(evandez): Uncomment once available.
            # 'biggan/imagenet',
        ),
        (
            'alexnet/places365',
            'resnet152/places365',
            # TODO(evandez): Uncomment once available.
            # 'biggan/places365',
        ),
    ),
    # TODO(evandez): Uncomment once available.
    # 'across-task': (
    #     (
    #         'alexnet/imagenet',
    #         'alexnet/places365',
    #         'resnet152/imagenet',
    #         'resnet152/places365',
    #     ),
    #     ('biggan/imagenet', 'biggan/places365'),
    # ),
}

SAT = 'sat'
SAT_MF = 'sat+mf'
SAT_WF = 'sat+wf'
SAT_MF_WF = 'sat+mf+wf'
MODELS = (SAT, SAT_MF, SAT_WF, SAT_MF_WF)

parser = argparse.ArgumentParser(description='run captioner abalations')
parser.add_argument('--experiments',
                    nargs='+',
                    help='experiments to run (default: all experiments)')
parser.add_argument('--models',
                    choices=MODELS,
                    nargs='+',
                    default=(SAT_MF_WF,),
                    help='captioning model to use')
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

# Load all featurizers in advance to avoid issues with reading model files
# after Kerberos tickets expire.
masked_image_featurizer = featurizers.MaskedImagePyramidFeaturizer().to(device)
masked_feats_featurizer = featurizers.MaskedPyramidFeaturizer().to(device)
featurizers_by_model = {
    SAT: masked_image_featurizer,
    SAT_MF: masked_feats_featurizer,
    SAT_MF_WF: masked_feats_featurizer,
}

for model in args.models:
    featurizer: Optional[featurizers.Featurizer]
    featurizer = featurizers_by_model.get(model)
    for experiment in args.experiments or EXPERIMENTS.keys():
        print(f'-------- BEGIN EXPERIMENT: {model}/{experiment} --------')

        # Have to handle within-network and across-* experiments differently.
        splits = EXPERIMENTS[experiment]
        if len(splits) == 2:
            left = zoo.datasets(*splits[0], path=args.datasets_root)
            right = zoo.datasets(*splits[1], path=args.datasets_root)
            configs = [(left, right, *splits),
                       (right, left, *reversed(splits))]
        else:
            assert len(splits) == 1, 'weird splits?'
            names, = splits
            configs = []
            for name in names:
                dataset = zoo.datasets(name, path=args.datasets_root)
                size = len(cast(Sized, dataset))
                test_size = int(.1 * size)
                train_size = size - test_size
                train, test = data.random_split(dataset,
                                                (train_size, test_size))
                configs.append((train, test, (name,), (name,)))

        # For every train/test set, train the captioner, test it, and log.
        for train_dataset, test_dataset, train_keys, test_keys in configs:
            train_features, test_features = None, None
            if featurizer is not None:
                train_features = featurizer.map(
                    cast(data.Dataset, train_dataset),
                    display_progress_as='featurize train set',
                    device=device)
                test_features = featurizer.map(
                    cast(data.Dataset, test_dataset),
                    display_progress_as='featurize test set',
                    device=device)

            annotator, annotator_f1 = None, None
            if args.model not in (SAT, SAT_MF):
                annotator = annotators.WordAnnotator.fit(
                    train, featurizer, features=train_features, device=device)
                annotator_f1, _ = annotator.score(
                    test,
                    features=test_features,
                    device=device,
                    display_progress_as='test word annotator')

            if args.model in (SAT, SAT_MF):
                assert featurizer is not None
                captioner = captioners.decoder(train, featurizer=featurizer)
            elif args.model == SAT_WF:
                assert annotator is not None
                assert featurizer is None
                captioner = captioners.decoder(train, annotator=annotator)
            else:
                assert args.model == SAT_MF_WF
                assert featurizer is not None
                assert annotator is not None
                captioner = captioners.decoder(train,
                                               featurizer=featurizer,
                                               annotator=annotator)

            # Train the model.
            captioner.fit(train, features=train_features, device=device)
            predictions = captioner.predict(test,
                                            features=test_features,
                                            device=device)
            bleu = captioner.bleu(test, predictions=predictions)
            rouge = captioner.rouge(test, predictions=predictions)

            # Log ALL the things!
            log = {
                'model': model,
                'experiment': experiment,
                'train': train_keys,
                'test': test_keys,
                'bleu': bleu.score,
            }
            if annotator_f1 is not None:
                log['annotator-f1'] = annotator_f1
            for index, precision in enumerate(bleu.precisions):
                log[f'bleu-{index + 1}'] = precision
            for kind, scores in rouge.items():
                for key, score in scores.items():
                    log[f'{kind}-{key}'] = score
            log['samples'] = [
                wandb.Image(
                    test[index].as_pil_image_grid(),
                    caption=f'({experiment.upper()}) {predictions[index]}',
                ) for index in range(min(len(test), args.wandb_n_samples))
            ]
            wandb.log(log)
