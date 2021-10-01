"""Run experiments for baseline table in paper."""
import argparse
import csv
import json
import pathlib
import re
import shutil

from lv import datasets, models, zoo
from lv.deps.ext import bert_score
from lv.utils import env, metrics
from lv.utils.typing import StrSequence

import wandb
from torch import cuda

ALEXNET_IMAGENET_REMAP = {
    'conv1': 'features-0',
    'conv2': 'features-3',
    'conv3': 'features-6',
    'conv4': 'features-8',
    'conv5': 'features-10',
}

EXPERIMENTS = (
    zoo.KEYS.ALEXNET_IMAGENET,
    zoo.KEYS.ALEXNET_PLACES365,
    zoo.KEYS.RESNET152_IMAGENET,
    zoo.KEYS.RESNET152_PLACES365,
)

METHOD_NETDISSECT = 'netdissect'
METHOD_COMPEXP = 'compexp'
METHOD_NO_PMI = 'no-pmi'
METHOD_PMI = 'pmi'
METHODS = (
    METHOD_NETDISSECT,
    METHOD_COMPEXP,
    METHOD_NO_PMI,
    METHOD_PMI,
)

parser = argparse.ArgumentParser(description='run captioner baselines')
parser.add_argument('--experiments',
                    nargs='+',
                    choices=EXPERIMENTS,
                    default=EXPERIMENTS,
                    help='experiments to run (default: all)')
parser.add_argument('--methods',
                    nargs='+',
                    choices=METHODS,
                    default=METHODS,
                    help='methods to run (default: all)')
parser.add_argument(
    '--netdissect-results-dir',
    type=pathlib.Path,
    help='netdissect results dir (default: project results dir)')
parser.add_argument('--compexp-results-dir',
                    type=pathlib.Path,
                    help='compexp results dir (default: project results dir)')
parser.add_argument('--trials',
                    type=int,
                    default=5,
                    help='repeat each experiment this many times (default: 5)')
parser.add_argument('--precompute-features',
                    action='store_true',
                    help='precompute visual features (default: do not)')
parser.add_argument('--results-dir',
                    type=pathlib.Path,
                    help='root dir for intermediate and final results '
                    '(default: project results dir)')
parser.add_argument('--clear-results-dir',
                    action='store_true',
                    help='if set, clear results dir (default: do not)')
parser.add_argument('--wandb-project',
                    default='lv',
                    help='wandb project name (default: lv)')
parser.add_argument('--wandb-name',
                    default='captioner-generalization',
                    help='wandb run name (default: captioner-generalization)')
parser.add_argument('--wandb-group',
                    default='captioner',
                    help='wandb group name (default: captioner)')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

wandb.init(project=args.wandb_project,
           name=args.wandb_name,
           group=args.wandb_group,
           config={'trials': args.trials})

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

results_dir = args.results_dir or (env.results_dir() / 'captioner-baselines')
if args.clear_results_dir and results_dir.exists():
    shutil.rmtree(results_dir)
results_dir.mkdir(exist_ok=True, parents=True)

netdissect_results_dir = args.netdissect_results_dir
if netdissect_results_dir is None:
    netdissect_results_dir = env.results_dir() / 'netdissect'

compexp_results_dir = args.compexp_results_dir
if compexp_results_dir is None:
    compexp_results_dir = env.results_dir() / 'compexp'

bert_scorer = bert_score.BERTScorer(lang='en',
                                    idf=True,
                                    rescale_with_baseline=True,
                                    use_fast_tokenizer=True,
                                    device=device)

for experiment in args.experiments:
    experiment_key = experiment.replace('/', '-')

    test = zoo.dataset(experiment)
    assert isinstance(test, datasets.AnnotatedTopImagesDataset)

    for method in args.methods:
        train = None
        if method in {METHOD_NO_PMI, METHOD_PMI}:
            train_group_key = f'NOT_{experiment_key.upper()}'
            train = zoo.datasets(*zoo.DATASET_GROUPINGS[train_group_key])

        trials = args.trials in method in {METHOD_PMI, METHOD_NO_PMI} else 1
        for trial in range(trials):
            if trials == 1:
                print(f'---- {experiment_key}/{method} ----')
            else:
                print(f'---- {experiment_key}/{method}/trial {trial} ----')

            predictions: StrSequence
            if method == METHOD_NETDISSECT:
                results_by_layer_unit = {}
                for layer in test.layers:
                    results_name = (f'{experiment_key}-netpqc-conv1-10/'
                                    'report.json')
                    results_file = netdissect_results_dir / results_name
                    with results_file.open('r') as handle:
                        results = json.load(handle)
                    for result in results['units']:
                        unit = str(result['unit'])
                        results_by_layer_unit[str(layer),
                                              unit] = result['label']

                predictions = []
                for index in range(len(test)):
                    sample = test[index]
                    layer = str(sample.layer)
                    unit = str(sample.unit)
                    predictions.append(results_by_layer_unit[layer, unit])

            elif method == METHOD_COMPEXP:
                model, dataset = experiment.split('/')
                model_subdir = f'{model}_{dataset}_broden_ade20k_neuron_3'
                results_by_layer_unit = {}
                for layer in test.layers:
                    layer_key = layer
                    if experiment == zoo.KEYS.ALEXNET_IMAGENET:
                        layer_key = ALEXNET_IMAGENET_REMAP[str(layer)]
                    results_file = (compexp_results_dir / model_subdir /
                                    f'tally_{layer_key}.csv')
                    with results_file.open('r') as handle:
                        rows = tuple(csv.DictReader(handle))
                    for row in rows:
                        unit = str(row['unit'])
                        label = row['label']

                        label = label.lower()\
                            .replace('(', '')\
                            .replace(')', '')\
                            .replace('-', ' ')\
                            .replace('_', ' ')
                        label = re.sub(r'\W+(s|t|c)($|\W+)', ' ', label)
                        label = label.strip()

                        results_by_layer_unit[str(layer), unit] = label

                predictions = []
                for index in range(len(test)):
                    sample = test[index]
                    layer = str(sample.layer)
                    unit = str(sample.unit)
                    predictions.append(results_by_layer_unit[layer, unit])

            else:
                assert method in {METHOD_NO_PMI, METHOD_PMI}
                assert train is not None

                trial_key = f'{experiment_key}-trial{trial}'
                captioner_file = results_dir / f'{trial_key}-captioner.pth'
                if captioner_file.exists():
                    print(f'loading captioner from {captioner_file}')
                    decoder = models.Decoder.load(captioner_file,
                                                  map_location=device)
                else:
                    lm = None
                    if method == METHOD_PMI:
                        lm = models.lm(train)
                        lm.fit(train, device=device)

                    encoder = models.encoder()

                    train_features = None
                    if args.precompute_features:
                        train_features = encoder.map(train, device=device)

                    decoder = models.decoder(train, encoder, lm=lm)
                    decoder.fit(train, features=train_features, device=device)

                    print(f'saving decoder to {captioner_file}')
                    decoder.save(captioner_file)

                predictions = decoder.predict(test,
                                              strategy='beam',
                                              beam_size=50,
                                              temperature=.2,
                                              mi=method == METHOD_PMI,
                                              device=device)

            bert_scores = metrics.bert_score(test,
                                             predictions,
                                             bert_scorer=bert_scorer)
            bleu = metrics.bleu(test, predictions)

            log = {'experiment': experiment, 'method': method, 'trial': trial}
            for index, precision in enumerate(bleu.precisions):
                log[f'bleu-{index + 1}'] = precision
            for kind, score in bert_scores.items():
                log[f'bert_score-{kind}'] = score

            wandb.log(log)
