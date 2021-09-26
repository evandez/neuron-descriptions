"""Run captioner generalization experiments."""
import argparse
import pathlib
import shutil
from typing import Mapping, NamedTuple, Tuple

from lv import models, zoo
from lv.deps.ext import bert_score
from lv.utils import env, training, viz
from lv.utils.typing import StrSequence

import torch
import wandb
from torch import cuda
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
EXPERIMENT_ACROSS_ARCH = 'across-arch'
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
    EXPERIMENT_ACROSS_ARCH: (
        (
            'alexnet/imagenet',
            'alexnet/places365',
            'resnet152/imagenet',
            'resnet152/places365',
            'biggan/imagenet',
            'biggan/places365',
        ),
        ('dino_vits8/imagenet',),
    )
}

parser = argparse.ArgumentParser(
    description='run captioner generalization experiments')
parser.add_argument('--experiments',
                    nargs='+',
                    help='experiments to run (default: all experiments)')
parser.add_argument('--trials',
                    type=int,
                    default=5,
                    help='repeat each experiment this many times (default: 5)')
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
parser.add_argument('--wandb-n-samples',
                    type=int,
                    default=25,
                    help='number of samples to upload for each model')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

wandb.init(project=args.wandb_project,
           name=args.wandb_name,
           group=args.wandb_group,
           config={'trials': args.trials})
run = wandb.run
assert run is not None, 'failed to initialize wandb?'

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

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
encoder = models.encoder().to(device)

# Start experiments.
for experiment in args.experiments or EXPERIMENTS.keys():
    print(f'\n-------- BEGIN EXPERIMENT: {experiment} --------')

    # Have to handle within-network and across-* experiments differently.
    splits = EXPERIMENTS[experiment]
    if len(splits) == 2:
        left = zoo.datasets(*splits[0], path=data_dir)
        right = zoo.datasets(*splits[1], path=data_dir)
        configs = [LoadedSplit(left, right, *splits)]
        if experiment != EXPERIMENT_ACROSS_ARCH:
            configs += [LoadedSplit(right, left, *reversed(splits))]
    else:
        assert experiment == EXPERIMENT_WITHIN_NETWORK
        assert len(splits) == 1
        names, = splits
        configs = []
        for name in names:
            dataset = zoo.datasets(name, path=data_dir)
            splits_file = results_dir / f'{name.replace("/", "_")}-splits.pth'
            if splits_file.exists():
                print(f'loading {name} w/i-network splits from {splits_file}')
                indices = torch.load(splits_file)['test']
                split = training.fixed_split(dataset, indices)
            else:
                split = training.random_split(dataset, hold_out=args.hold_out)
                print(f'saving {name} w/i-network splits to {splits_file}')
                torch.save(
                    {
                        'train': split[0].indices,
                        'test': split[1].indices,
                    }, splits_file)
            configs.append(LoadedSplit(*split, (name,), (name,)))

    # For every train/test set, train the captioner, test it, and log.
    for index, (train, test, train_keys, test_keys) in enumerate(configs):
        assert isinstance(train, data.Dataset)
        assert isinstance(test, data.Dataset)

        # Maybe precompute image features.
        train_features, test_features = None, None
        if args.precompute_features:
            train_features = encoder.map(train, device=device)
            test_features = encoder.map(test, device=device)

        for trial in range(args.trials):
            trial_key = f'{experiment}-split{index}-trial{trial}'

            # Train the LM.
            lm_file = results_dir / f'{trial_key}-lm.pth'
            if lm_file.exists():
                print(f'loading lm from {lm_file}')
                lm = models.LanguageModel.load(lm_file, map_location=device)
            else:
                lm = models.lm(train)
                lm.fit(train, device=device)
                print(f'saving lm to {lm_file}')
                lm.save(lm_file)

            # Train the decoder.
            decoder_file = results_dir / f'{trial_key}-captioner.pth'
            if decoder_file.exists():
                print(f'loading decoder from {decoder_file}')
                decoder = models.Decoder.load(decoder_file,
                                              map_location=device)
                decoder.eval()
            else:
                decoder = models.decoder(train,
                                         encoder,
                                         lm=lm,
                                         strategy='rerank',
                                         beam_size=50,
                                         temperature=.2)
                decoder.fit(train, features=train_features, device=device)
                print(f'saving decoder to {decoder_file}')
                decoder.save(decoder_file)

            # Test the decoder.
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
                'trial': trial,
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
            log['samples'] = viz.random_neuron_wandb_images(
                test,
                captions=predictions,
                k=args.wandb_n_samples,
                experiment=experiment,
                trial=trial,
                train=tuple(train_keys),
                test=tuple(test_keys))
            wandb.log(log)
