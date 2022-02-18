"""Run MILAN generalization experiments."""
import argparse
import pathlib
import shutil
from typing import Mapping, NamedTuple, Tuple

from src import milan, milannotations
from src.deps.ext import bert_score
from src.utils import env, training, viz
from src.utils.typing import StrSequence

import torch
import wandb
from torch import cuda
from torch.utils import data


class LoadedSplit(NamedTuple):
    """Wrap a loaded train/test split and its metadata."""

    train: data.Dataset
    test: data.Dataset
    train_key: str
    test_key: str


DatasetNames = StrSequence
Splits = Tuple[DatasetNames, ...]

EXPERIMENT_WITHIN_NETWORK = 'within-network'
EXPERIMENT_ACROSS_NETWORK = 'across-network'
EXPERIMENT_ACROSS_DATASET = 'across-dataset'
EXPERIMENT_ACROSS_TASK = 'across-task'
EXPERIMENT_ACROSS_ARCH = 'across-arch'
EXPERIMENT_LEAVE_ONE_OUT = 'leave-one-out'
EXPERIMENTS: Mapping[str, Splits] = {
    EXPERIMENT_WITHIN_NETWORK: (
        milannotations.KEYS.ALEXNET_IMAGENET,
        milannotations.KEYS.ALEXNET_PLACES365,
        milannotations.KEYS.RESNET152_IMAGENET,
        milannotations.KEYS.RESNET152_PLACES365,
        milannotations.KEYS.BIGGAN_IMAGENET,
        milannotations.KEYS.BIGGAN_PLACES365,
    ),
    EXPERIMENT_ACROSS_NETWORK: ((
        milannotations.KEYS.ALEXNET,
        milannotations.KEYS.RESNET152,
    ),),
    EXPERIMENT_ACROSS_DATASET: ((
        milannotations.KEYS.IMAGENET,
        milannotations.KEYS.PLACES365,
    ),),
    EXPERIMENT_ACROSS_TASK: ((
        milannotations.KEYS.CLASSIFIERS,
        milannotations.KEYS.GENERATORS,
    ),),
    EXPERIMENT_ACROSS_ARCH: ((
        milannotations.KEYS.BASE,
        milannotations.KEYS.DINO_VITS8_IMAGENET,
    ),),
    EXPERIMENT_LEAVE_ONE_OUT: (
        (
            milannotations.KEYS.NOT_ALEXNET_IMAGENET,
            milannotations.KEYS.ALEXNET_IMAGENET,
        ),
        (
            milannotations.KEYS.NOT_ALEXNET_PLACES365,
            milannotations.KEYS.ALEXNET_PLACES365,
        ),
        (
            milannotations.KEYS.NOT_RESNET152_IMAGENET,
            milannotations.KEYS.RESNET152_IMAGENET,
        ),
        (
            milannotations.KEYS.NOT_RESNET152_PLACES365,
            milannotations.KEYS.RESNET152_PLACES365,
        ),
        (
            milannotations.KEYS.NOT_BIGGAN_IMAGENET,
            milannotations.KEYS.BIGGAN_IMAGENET,
        ),
        (
            milannotations.KEYS.NOT_BIGGAN_PLACES365,
            milannotations.KEYS.BIGGAN_PLACES365,
        ),
    ),
}

parser = argparse.ArgumentParser(description='run generalization experiments')
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
                    help='root dir for all results '
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
                    default='milan',
                    help='wandb project name (default: milan)')
parser.add_argument('--wandb-name',
                    default='generalization',
                    help='wandb run name (default: generalization)')
parser.add_argument('--wandb-group',
                    default='experiments',
                    help='wandb group name (default: experiments)')
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
results_dir = args.results_dir or (env.results_dir() / 'generalization')
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
encoder = milan.encoder().to(device)

# Start experiments.
for experiment in args.experiments or EXPERIMENTS.keys():
    print(f'\n-------- BEGIN EXPERIMENT: {experiment} --------')

    # Have to handle within-network and across-* experiments differently.
    splits = EXPERIMENTS[experiment]
    if isinstance(splits[0], tuple):
        configs = []
        for left_key, right_key in splits:
            left = milannotations.load(left_key, path=data_dir)
            right = milannotations.load(right_key, path=data_dir)
            config = LoadedSplit(left, right, left_key, right_key)
            configs.append(config)
            if experiment != EXPERIMENT_ACROSS_ARCH:
                config = LoadedSplit(right, left, right_key, left_key)
                configs.append(config)
    else:
        assert experiment == EXPERIMENT_WITHIN_NETWORK
        configs = []
        for name in splits:
            assert isinstance(name, str), name
            dataset = milannotations.load(name, path=data_dir)
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

    # For every train/test set, train the decoder, test it, and log.
    for split_id, (train, test, train_keys, test_keys) in enumerate(configs):
        assert isinstance(train, data.Dataset)
        assert isinstance(test, data.Dataset)

        # Maybe precompute image features.
        train_features, test_features = None, None
        if args.precompute_features:
            train_features = encoder.map(train, device=device)
            test_features = encoder.map(test, device=device)

        for trial_id in range(args.trials):
            trial_key = f'{experiment}-split{split_id}-trial{trial_id}'

            # Train the LM.
            lm_file = results_dir / f'{trial_key}-lm.pth'
            if lm_file.exists():
                print(f'loading lm from {lm_file}')
                lm = milan.LanguageModel.load(lm_file, map_location=device)
            else:
                lm = milan.lm(train)
                lm.fit(train, device=device)
                print(f'saving lm to {lm_file}')
                lm.save(lm_file)

            # Train the decoder.
            decoder_file = results_dir / f'{trial_key}-decoder.pth'
            if decoder_file.exists():
                print(f'loading decoder from {decoder_file}')
                decoder = milan.Decoder.load(decoder_file, map_location=device)
                decoder.eval()
            else:
                decoder = milan.decoder(train,
                                        encoder,
                                        lm=lm,
                                        strategy='rerank',
                                        beam_size=50,
                                        temperature=.2)
                decoder.fit(train,
                            features=train_features,
                            patience=10
                            if experiment == EXPERIMENT_WITHIN_NETWORK else 4,
                            device=device)
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
                'trial': trial_id,
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
                trial=trial_id,
                train=tuple(train_keys),
                test=tuple(test_keys))
            wandb.log(log)
