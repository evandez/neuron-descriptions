"""Run experiments for baseline table in paper."""
import argparse
import csv
import json
import pathlib
import re
import shutil

from src import milan, milannotations
from src.deps.ext import bert_score
from src.utils import env, metrics
from src.utils.typing import StrSequence

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
    milannotations.KEYS.ALEXNET_IMAGENET,
    milannotations.KEYS.ALEXNET_PLACES365,
    milannotations.KEYS.RESNET152_IMAGENET,
    milannotations.KEYS.RESNET152_PLACES365,
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

parser = argparse.ArgumentParser(description='run baselines')
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
                    default='milan',
                    help='wandb project name (default: milan)')
parser.add_argument('--wandb-name',
                    default='baselines',
                    help='wandb run name (default: baselines)')
parser.add_argument('--wandb-group',
                    default='experiments',
                    help='wandb group name (default: experiments)')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

wandb.init(project=args.wandb_project,
           name=args.wandb_name,
           group=args.wandb_group,
           config={'trials': args.trials})

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

results_dir = args.results_dir or (env.results_dir() / 'baselines')
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

    test = milannotations.load(experiment)
    assert isinstance(test, milannotations.AnnotatedTopImagesDataset)

    for method in args.methods:
        train = None
        if method in {METHOD_NO_PMI, METHOD_PMI}:
            train_group_key = f'not-{experiment_key}'
            train = milannotations.load(train_group_key)

        trials = args.trials if method in {METHOD_PMI, METHOD_NO_PMI} else 1
        for trial in range(trials):
            if trials == 1:
                print(f'---- {experiment_key}/{method} ----')
            else:
                print(f'---- {experiment_key}/{method}/trial {trial} ----')

            predictions: StrSequence
            if method == METHOD_NETDISSECT:
                results_by_layer_unit = {}
                for layer in test.layers:
                    results_name = (f'{experiment_key.replace("365", "")}'
                                    f'-netpqc-{layer}-10/report.json')
                    results_file = netdissect_results_dir / results_name
                    with results_file.open('r') as handle:
                        results = json.load(handle)
                    for result in results['units']:
                        unit = str(result['unit'])
                        label = result['label'].split('-')[0]
                        results_by_layer_unit[str(layer), unit] = label

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
                    if experiment == milannotations.KEYS.ALEXNET_IMAGENET:
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

                milan_key = f'{experiment_key}-trial{trial}'
                milan_file = results_dir / f'{milan_key}-captioner.pth'
                if milan_file.exists():
                    print(f'loading decoder from {milan_file}')
                    decoder = milan.Decoder.load(milan_file,
                                                 map_location=device)
                else:
                    lm_file = results_dir / f'{milan_key}-lm.pth'
                    if lm_file.exists():
                        print(f'loading lm from {lm_file}')
                        lm = milan.LanguageModel.load(lm_file,
                                                      map_location=device)
                    else:
                        lm = milan.lm(train)
                        lm.fit(train, device=device)
                        print(f'saving lm to {lm_file}')
                        lm.save(lm_file)

                    encoder = milan.encoder()

                    train_features = None
                    if args.precompute_features:
                        train_features = encoder.map(train, device=device)

                    decoder = milan.decoder(train, encoder, lm=lm)
                    decoder.fit(train, features=train_features, device=device)

                    print(f'saving decoder to {milan_file}')
                    decoder.save(milan_file)

                decoder.eval()
                predictions = decoder.predict(
                    test,
                    strategy='rerank' if method == METHOD_PMI else 'greedy',
                    beam_size=50,
                    temperature=.2,
                    mi=False,
                    device=device)

            # Save the predictions.
            outputs = [('layer', 'unit', 'description')]
            for index in range(len(test)):
                sample = test[index]
                output = (sample.layer, str(sample.unit), predictions[index])
                outputs.append(output)
            trial_key = f'{experiment_key}-{method}-{trial}'
            captions_file = results_dir / f'{trial_key}-descriptions.csv'
            with captions_file.open('w') as handle:
                csv.writer(handle).writerows(outputs)

            # Compute metrics.
            bert_scores = metrics.bert_score(test,
                                             predictions,
                                             bert_scorer=bert_scorer)
            bleu = metrics.bleu(test, predictions)

            log = {'experiment': experiment, 'method': method, 'trial': trial}

            log['bleu'] = bleu.score
            for index, precision in enumerate(bleu.precisions):
                log[f'bleu-{index + 1}'] = precision

            for kind, score in bert_scores.items():
                log[f'bert_score-{kind}'] = score

            wandb.log(log)
