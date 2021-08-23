"""Run a series of ablations on the captioning model."""
import argparse
import pathlib
import shutil
from typing import Any

from lv import zoo
from lv.ext import bert_score
from lv.models import decoders, encoders, lms
from lv.utils import env, logging, training

import numpy
import torch
import wandb
from torch.utils import data

ABLATION_BASE = 'base'
ABLATION_BEAM = 'beam'
ABLATION_MI = 'mi'
ABLATION_BEAM_MI = 'beam-mi'
ABLATIONS = (
    ABLATION_BASE,
    ABLATION_BEAM,
    ABLATION_MI,
    ABLATION_BEAM_MI,
)

DATASETS = (
    zoo.KEY_ALEXNET_IMAGENET,
    zoo.KEY_ALEXNET_PLACES365,
    zoo.KEY_RESNET152_IMAGENET,
    zoo.KEY_RESNET152_PLACES365,
    # zoo.KEY_BIGGAN_IMAGENET,
    # zoo.KEY_BIGGAN_PLACES365,
)

ENCODER_RESNET18 = 'resnet18'
ENCODER_RESNET50 = 'resnet50'
ENCODERS = (ENCODER_RESNET18, ENCODER_RESNET50)

SCORE_BLEU = 'bleu'
SCORE_ROUGE = 'rouge'
SCORE_BERT_SCORE = 'bert-score'
SCORES = (
    SCORE_BLEU,
    SCORE_ROUGE,
    SCORE_BERT_SCORE,
)

parser = argparse.ArgumentParser(description='ablate and evaluate captioner')
parser.add_argument('--ablations',
                    default=ABLATIONS,
                    nargs='+',
                    help='ablations to run (default: all)')
parser.add_argument('--datasets',
                    default=DATASETS,
                    nargs='+',
                    help='datasets to train/test on (default: all)')
parser.add_argument('--encoders',
                    default=ENCODERS,
                    nargs='+',
                    help='encoders to try training decoder on (default: all)')
parser.add_argument('--scores',
                    nargs='+',
                    default=SCORES,
                    help='scores to compute (default: all)')
parser.add_argument(
    '--hold-out',
    type=float,
    default=.1,
    help='hold out and test on this fraction of data (default: .1)')
parser.add_argument('--precompute-features',
                    action='store_true',
                    help='precompute visual features (default: do not)')
parser.add_argument(
    '--beam-size-min',
    type=int,
    default=5,
    help='min temperature to try in mi ablations (default: .05)')
parser.add_argument(
    '--beam-size-max',
    type=int,
    default=25,
    help='max temperature to try in mi ablations (default: .3)')
parser.add_argument(
    '--beam-size-step',
    type=int,
    default=5,
    help='step size for temperatures to try in mi ablations (default: .05)')
parser.add_argument(
    '--mi-temperature-min',
    type=float,
    default=.05,
    help='min temperature to try in mi ablations (default: .05)')
parser.add_argument(
    '--mi-temperature-max',
    type=float,
    default=.25,
    help='max temperature to try in mi ablations (default: .3)')
parser.add_argument(
    '--mi-temperature-step',
    type=float,
    default=.025,
    help='step size for temperatures to try in mi ablations (default: .05)')
parser.add_argument('--data-dir',
                    type=pathlib.Path,
                    help='root dir for datasets (default: project data dir)')
parser.add_argument('--results-dir',
                    type=pathlib.Path,
                    help='directory to write intermediate and final results '
                    '(default: <project results dir>/captioner-ablations)')
parser.add_argument('--clear-results-dir',
                    action='store_true',
                    help='if set, clear results dir (default: do not)')
parser.add_argument('--wandb-project',
                    default='lv',
                    help='wandb project name (default: lv)')
parser.add_argument('--wandb-name',
                    default='captioner-ablations',
                    help='wandb run name (default: captioner-ablations)')
parser.add_argument('--wandb-group',
                    default='captioner',
                    help='wandb group name (default: captioner)')
parser.add_argument('--wandb-entity',
                    help='wandb user or team (default: wandb default)')
parser.add_argument('--wandb-dir',
                    metavar='PATH',
                    help='wandb directory (default: wandb default)')
parser.add_argument(
    '--wandb-n-samples',
    type=int,
    default=10,
    help='number of samples to upload for each model (default: 10)')
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
results_dir = args.results_dir or (env.results_dir() / 'captioner-ablations')
if args.clear_results_dir and results_dir.exists():
    shutil.rmtree(results_dir)
results_dir.mkdir(exist_ok=True, parents=True)

bert_scorer = None
if SCORE_BERT_SCORE in args.scores:
    bert_scorer = bert_score.BERTScorer(lang='en',
                                        idf=True,
                                        rescale_with_baseline=True,
                                        use_fast_tokenizer=True,
                                        device=device)

dataset = zoo.datasets(*args.datasets)
for config in args.encoders:
    print(f'---- begin ablations for {config} encoder ----')

    splits_file = results_dir / f'{config}-splits.pth'
    if splits_file.exists():
        print(f'loading cached train/test splits from {splits_file}')
        splits = torch.load(splits_file)
        train = data.Subset(dataset, splits['train'])
        test = data.Subset(dataset, splits['test'])
    else:
        train, test = training.random_split(dataset, hold_out=args.hold_out)
        print(f'writing train/test splits to {splits_file}')
        torch.save({'train': train.indices, 'test': test.indices}, splits_file)

    lm = None
    lm_file = results_dir / f'{config}-lm.pth'
    if lm_file.exists():
        print(f'loading cached lm from {lm_file}')
        lm = lms.LanguageModel.load(lm_file, map_location=device)
    elif ABLATION_MI in args.ablations or ABLATION_BEAM_MI in args.ablations:
        lm = lms.lm(train).to(device)
        lm.fit(train,
               device=device,
               display_progress_as=f'(encoder={config}) train lm')
        print(f'saving lm to {lm_file}')
        lm.save(lm_file)

    captioner_file = results_dir / f'{config}-captioner.pth'
    if captioner_file.is_file() and splits_file.is_file():
        print(f'loading cached captioner from {captioner_file}')
        decoder = decoders.Decoder.load(captioner_file, map_location=device)
        encoder = decoder.encoder
    else:
        encoder = encoders.PyramidConvEncoder(config=config).to(device)
        decoder = decoders.decoder(train, encoder, lm=lm).to(device)

        train_features, test_features = None, None
        if args.precompute_features:
            train_features = encoder.map(
                train,
                device=device,
                display_progress_as=f'(encoder={config}) featurize train set')
            test_features = encoder.map(
                test,
                device=device,
                display_progress_as=f'(encoder={config}) featurize test set')

        decoder.fit(train,
                    features=train_features,
                    display_progress_as=f'(encoder={config}) train decoder',
                    device=device)

        print(f'saving captioner to {captioner_file}')
        decoder.save(captioner_file)

    def evaluate(**kwargs: Any) -> None:
        """Evaluate the captioner with the given args."""
        metadata = logging.kwargs_to_str(**kwargs)
        predictions = decoder.predict(
            test,
            features=test_features,
            device=device,
            display_progress_as=f'({metadata}) predict captions',
            **kwargs)

        log = {
            'encoder': config,
            'condition': kwargs,
        }
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

        log['samples'] = logging.random_neuron_wandb_images(
            test,
            captions=predictions,
            k=args.wandb_n_samples,
            **kwargs,
        )

        wandb.log(log)

    for ablation in args.ablations:
        if ablation == ABLATION_BASE:
            evaluate(strategy='greedy', mi=False)
        elif ablation == ABLATION_BEAM:
            for beam_size in numpy.arange(args.beam_size_min,
                                          args.beam_size_max,
                                          args.beam_size_step):
                evaluate(strategy='beam', mi=False, beam_size=beam_size)
        elif ablation == ABLATION_MI:
            for temperature in numpy.arange(args.mi_temperature_min,
                                            args.mi_temperature_max,
                                            args.mi_temperature_step):
                evaluate(strategy='greedy', mi=True, temperature=temperature)
        else:
            assert ablation == ABLATION_BEAM_MI
            for beam_size in numpy.arange(args.beam_size_min,
                                          args.beam_size_max,
                                          args.beam_size_step):
                for temperature in numpy.arange(args.mi_temperature_min,
                                                args.mi_temperature_max,
                                                args.mi_temperature_step):
                    evaluate(strategy='beam',
                             beam_size=beam_size,
                             mi=True,
                             temperature=temperature)
