"""Run captioner generalization experiments."""
import argparse
import pathlib
import shutil
from typing import Dict, Mapping, NamedTuple, Optional, Tuple

from lv import zoo
from lv.ext import bert_score
from lv.models import decoders, encoders, lms
from lv.utils import env, logging, training
from lv.utils.typing import StrSequence

import wandb
from torch.utils import data


class LoadedSplit(NamedTuple):
    """Wrap a loaded train/test split and its metadata."""

    train: data.Dataset
    test: data.Dataset
    train_keys: StrSequence
    test_keys: StrSequence


DatasetNames = StrSequence
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
parser.add_argument('--data-dir',
                    type=pathlib.Path,
                    help='root dir for datasets (default: project data dir)')
parser.add_argument('--results-dir',
                    type=pathlib.Path,
                    help='root dir for intermediate and final results '
                    '(default: project results dir)')
parser.add_argument('--clear-results-dir',
                    action='store_true',
                    help='if set, clear results dir (default: do not)')
parser.add_argument('--hold-out',
                    type=float,
                    default=.1,
                    help='hold out this fraction of data for testing')
parser.add_argument('--strategy',
                    default=decoders.STRATEGY_BEAM,
                    help='decoding strategy (default: beam)')
parser.add_argument(
    '--no-mi',
    action='store_true',
    help='if set, do not test with mi decoding (default: use mi decoding)')
parser.add_argument('--beam-size',
                    type=int,
                    default=20,
                    help='beam size for beam search decoding (default: 20)')
parser.add_argument('--temperature',
                    type=float,
                    default=.075,
                    help='mi decoding temperature (default: .075)')
parser.add_argument('--precompute-features',
                    action='store_true',
                    help='precompute visual features (default: do not)')
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

# Prepare necessary directories.
data_dir = args.data_dir or env.data_dir()
results_dir = args.results_dir or (env.results_dir() /
                                   'captioner-generalization')
if args.clear_results_dir and results_dir.exists():
    shutil.rmtree(results_dir)
results_dir.mkdir(exist_ok=True, parents=True)

# Load BERTScorer once up front.
bert_scorer = bert_score.BERTScorer(lang='en',
                                    idf=True,
                                    rescale_with_baseline=True,
                                    use_fast_tokenizer=True,
                                    device=device)

# Load encoder.
encoder = encoders.PyramidConvEncoder(config='resnet50').to(device)

# Start experiments.
for experiment in args.experiments or EXPERIMENTS.keys():
    print(f'\n-------- BEGIN EXPERIMENT: {experiment} --------')

    # Have to handle within-network and across-* experiments differently.
    splits = EXPERIMENTS[experiment]
    if len(splits) == 2:
        left = zoo.datasets(*splits[0], path=data_dir)
        right = zoo.datasets(*splits[1], path=data_dir)
        configs = [
            LoadedSplit(left, right, *splits),
            LoadedSplit(right, left, *reversed(splits)),
        ]
    elif experiment == EXPERIMENT_WITHIN_NETWORK:
        assert len(splits) == 1
        names, = splits
        configs = []
        for name in names:
            dataset = zoo.datasets(name, path=data_dir)
            split = training.random_split(dataset, hold_out=.1)
            configs.append(LoadedSplit(*split, (name,), (name,)))
    else:
        assert len(splits) == 1
        assert experiment == EXPERIMENT_LEAVE_ONE_OUT

        names, = splits

        datasets_by_name: Dict[str, data.Dataset] = {}
        for name in names:
            dataset = zoo.datasets(name, path=data_dir)
            datasets_by_name[name] = dataset

        unique = set(names)
        configs = []
        for name in unique:

            test_keys: StrSequence = (name,)
            test = datasets_by_name[name]

            train_keys: StrSequence = tuple(sorted(unique - {name}))
            assert train_keys

            train: Optional[data.Dataset] = None
            for other in train_keys:
                if train is None:
                    train = datasets_by_name[other]
                else:
                    assert train is not None
                    train += datasets_by_name[other]

            assert train is not None
            configs.append(LoadedSplit(train, test, train_keys, test_keys))

    # For every train/test set, train the captioner, test it, and log.
    for index, (train, test, train_keys, test_keys) in enumerate(configs):
        assert isinstance(train, data.Dataset)
        assert isinstance(test, data.Dataset)

        # Train the LM.
        lm = lms.lm(train)
        lm.fit(train, device=device)
        lm.save(results_dir / f'{experiment}-split{index}-lm.pth')

        # Maybe precompute image features.
        train_features, test_features = None, None
        if args.precompute_features:
            train_features = encoder.map(train, device=device)
            test_features = encoder.map(test, device=device)

        # Train the decoder.
        decoder = decoders.decoder(train,
                                   encoder,
                                   lm=lm,
                                   strategy=args.strategy,
                                   beam_size=args.beam_size,
                                   temperature=args.temperature)
        decoder.fit(train, features=train_features, device=device)
        decoder.save(results_dir / f'{experiment}-split{index}-captioner.pth')

        # Test the decoder.
        predictions = decoder.predict(test,
                                      mi=not args.no_mi,
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
        log['samples'] = logging.random_neuron_wandb_images(
            test,
            captions=predictions,
            k=args.wandb_n_samples,
            experiment=experiment,
            train=tuple(train_keys),
            test=tuple(test_keys))
        wandb.log(log)
