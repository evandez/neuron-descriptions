"""Run a series of sweeps on the captioning model."""
import argparse
import pathlib
import shutil
from typing import Any, Dict

from src import milan, milannotations
from src.deps.ext import bert_score
from src.utils import env, training, viz

import numpy
import torch
import wandb
from torch import cuda

SWEEP_GREEDY = 'greedy'
SWEEP_BEAM = 'beam'
SWEEP_GREEDY_MI = 'greedy-mi'
SWEEP_BEAM_MI = 'beam-mi'
SWEEP_RERANK = 'rerank'
SWEEPS = (
    SWEEP_GREEDY,
    SWEEP_BEAM,
    SWEEP_GREEDY_MI,
    SWEEP_BEAM_MI,
    SWEEP_RERANK,
)

DATASETS = (
    milannotations.KEYS.ALEXNET_IMAGENET,
    milannotations.KEYS.ALEXNET_PLACES365,
    milannotations.KEYS.RESNET152_IMAGENET,
    milannotations.KEYS.RESNET152_PLACES365,
    milannotations.KEYS.BIGGAN_IMAGENET,
    milannotations.KEYS.BIGGAN_PLACES365,
)

ENCODER_RESNET18 = 'resnet18'
ENCODER_RESNET50 = 'resnet50'
ENCODER_RESNET101 = 'resnet101'
ENCODERS = (ENCODER_RESNET18, ENCODER_RESNET50, ENCODER_RESNET101)

SCORE_BLEU = 'bleu'
SCORE_ROUGE = 'rouge'
SCORE_BERT_SCORE = 'bert-score'
SCORES = (
    SCORE_BLEU,
    SCORE_ROUGE,
    SCORE_BERT_SCORE,
)

parser = argparse.ArgumentParser(description='sweep over milan hyperparams')
parser.add_argument('--sweeps',
                    default=SWEEPS,
                    nargs='+',
                    help='sweeps to run (default: all)')
parser.add_argument('--dataset',
                    default=milannotations.KEYS.BASE,
                    help='milannotations to train/test on (default: base)')
parser.add_argument('--encoder',
                    choices=ENCODERS,
                    default=ENCODER_RESNET101,
                    help='encoder config (default: resnet101)')
parser.add_argument('--scores',
                    nargs='+',
                    default=SCORES,
                    help='scores to compute (default: all)')
parser.add_argument('--pretrained',
                    type=pathlib.Path,
                    help='path to results dir from run_milan_training.py; '
                    'if set, use this milan and its train/val splits')
parser.add_argument(
    '--hold-out',
    type=float,
    default=.1,
    help='hold out and test on this fraction of data (default: .1)')
parser.add_argument('--precompute-features',
                    action='store_true',
                    help='precompute visual features (default: do not)')
parser.add_argument('--beam-size-min',
                    type=int,
                    default=5,
                    help='min temperature to try in mi sweeps (default: .05)')
parser.add_argument('--beam-size-max',
                    type=int,
                    default=50,
                    help='max temperature to try in mi sweeps (default: .3)')
parser.add_argument(
    '--beam-size-step',
    type=int,
    default=5,
    help='step size for temperatures to try in mi sweeps (default: .05)')
parser.add_argument('--mi-temperature-min',
                    type=float,
                    default=.05,
                    help='min temperature to try in mi sweeps (default: .05)')
parser.add_argument('--mi-temperature-max',
                    type=float,
                    default=.75,
                    help='max temperature to try in mi sweeps (default: .3)')
parser.add_argument(
    '--mi-temperature-step',
    type=float,
    default=.05,
    help='step size for temperatures to try in mi sweeps (default: .05)')
parser.add_argument('--data-dir',
                    type=pathlib.Path,
                    help='root dir for datasets (default: project data dir)')
parser.add_argument('--results-dir',
                    type=pathlib.Path,
                    help='directory to write intermediate and final results '
                    '(default: <project results dir>/milan-<encoder>-sweep)')
parser.add_argument('--clear-results-dir',
                    action='store_true',
                    help='if set, clear results dir (default: do not)')
parser.add_argument('--wandb-project',
                    default='lv',
                    help='wandb project name (default: lv)')
parser.add_argument('--wandb-name',
                    help='wandb run name (default: milan-<encoder>-sweep)')
parser.add_argument('--wandb-group',
                    default='milan',
                    help='wandb group name (default: milan)')
parser.add_argument(
    '--wandb-n-samples',
    type=int,
    default=10,
    help='number of samples to upload for each model (default: 10)')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

config = args.encoder
key = f'milan-{config}-sweep'
wandb.init(project=args.wandb_project,
           name=args.wandb_name or key,
           group=args.wandb_group)
run = wandb.run
assert run is not None, 'failed to initialize wandb?'

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

# Prepare necessary directories.
data_dir = args.data_dir or env.data_dir()
results_dir = args.results_dir or (env.results_dir() / key)
if args.clear_results_dir and results_dir.exists():
    shutil.rmtree(results_dir)
results_dir.mkdir(exist_ok=True, parents=True)

# Import pretrained milan if necessary.
if args.pretrained:
    for child in args.pretrained.iterdir():
        shutil.copy(child, results_dir)

# Prepare BERT scorer.
bert_scorer = None
if SCORE_BERT_SCORE in args.scores:
    bert_scorer = bert_score.BERTScorer(lang='en',
                                        idf=True,
                                        rescale_with_baseline=True,
                                        use_fast_tokenizer=True,
                                        device=device)

dataset = milannotations.load(args.dataset, path=data_dir)

splits_file = results_dir / 'splits.pth'
if splits_file.exists():
    print(f'loading cached train/test splits from {splits_file}')
    splits = torch.load(splits_file)
    test, train = training.fixed_split(dataset, splits['train'])
else:
    train, test = training.random_split(dataset, hold_out=args.hold_out)
    print(f'writing train/test splits to {splits_file}')
    torch.save({'train': train.indices, 'test': test.indices}, splits_file)

lm = None
lm_file = results_dir / 'lm.pth'
if lm_file.exists():
    print(f'loading cached lm from {lm_file}')
    lm = milan.LanguageModel.load(lm_file, map_location=device)
elif {SWEEP_GREEDY_MI, SWEEP_BEAM_MI} & set(args.sweeps):
    lm = milan.lm(train).to(device)
    lm.fit(train, device=device, display_progress_as='train lm')
    print(f'saving lm to {lm_file}')
    lm.save(lm_file)

decoder_file = results_dir / 'decoder.pth'
if decoder_file.is_file() and splits_file.is_file():
    print(f'loading cached decoder from {decoder_file}')
    decoder = milan.Decoder.load(decoder_file, map_location=device).eval()
    encoder = decoder.encoder
else:
    encoder = milan.encoder(config=config).to(device)
    decoder = milan.decoder(train, encoder, lm=lm).to(device)

    train_features = None
    if args.precompute_features:
        train_features = encoder.map(train,
                                     device=device,
                                     display_progress_as='featurize train set')

    decoder.fit(train,
                features=train_features,
                display_progress_as='train decoder',
                device=device)

    print(f'saving decoder to {decoder_file}')
    decoder.save(decoder_file)

test_features = None
if args.precompute_features:
    test_features = encoder.map(test,
                                device=device,
                                display_progress_as='featurize test set')


def evaluate(**kwargs: Any) -> None:
    """Evaluate the milan with the given args."""
    assert isinstance(decoder, milan.Decoder)
    metadata = viz.kwargs_to_str(**kwargs)
    predictions = decoder.predict(
        test,
        features=test_features,
        device=device,
        display_progress_as=f'({metadata}) predict descriptions',
        **kwargs)

    log: Dict[str, Any] = {'condition': kwargs}
    if SCORE_BLEU in args.scores:
        bleu = decoder.bleu(test, predictions=predictions)
        log['bleu'] = bleu.score
        for index, precision in enumerate(bleu.precisions):
            log[f'bleu-{index + 1}'] = precision

    if SCORE_ROUGE in args.scores:
        rouge = decoder.rouge(test, predictions=predictions)
        for kind, scores in rouge.items():
            for key, score in scores.items():
                log[f'{kind}-{key}'] = score

    if SCORE_BERT_SCORE in args.scores:
        assert bert_scorer is not None
        bert_scores = decoder.bert_score(test,
                                         predictions=predictions,
                                         bert_scorer=bert_scorer)
        for kind, score in bert_scores.items():
            log[f'bert_score-{kind}'] = score

    log['samples'] = viz.random_neuron_wandb_images(
        test,
        captions=predictions,
        k=args.wandb_n_samples,
        **kwargs,
    )

    wandb.log(log)


for sweep in args.sweeps:
    if sweep == SWEEP_GREEDY:
        evaluate(strategy='greedy', mi=False)
    elif sweep == SWEEP_BEAM:
        for beam_size in numpy.arange(args.beam_size_min, args.beam_size_max,
                                      args.beam_size_step):
            evaluate(strategy='beam', mi=False, beam_size=beam_size)
    elif sweep == SWEEP_GREEDY_MI:
        for temperature in numpy.arange(args.mi_temperature_min,
                                        args.mi_temperature_max,
                                        args.mi_temperature_step):
            evaluate(strategy='greedy', mi=True, temperature=temperature)
    elif sweep == SWEEP_BEAM_MI:
        for beam_size in numpy.arange(args.beam_size_min, args.beam_size_max,
                                      args.beam_size_step):
            for temperature in numpy.arange(args.mi_temperature_min,
                                            args.mi_temperature_max,
                                            args.mi_temperature_step):
                evaluate(strategy='beam',
                         beam_size=beam_size,
                         mi=True,
                         temperature=temperature)
    else:
        assert sweep == SWEEP_RERANK
        for beam_size in numpy.arange(args.beam_size_min, args.beam_size_max,
                                      args.beam_size_step):
            for temperature in numpy.arange(args.mi_temperature_min,
                                            args.mi_temperature_max,
                                            args.mi_temperature_step):
                evaluate(strategy='rerank',
                         beam_size=beam_size,
                         temperature=temperature)
