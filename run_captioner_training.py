"""Train a neuron captioner from scratch."""
import argparse
import pathlib
import shutil
from typing import Optional

from lv import models, zoo
from lv.utils import env, training

import torch
from torch import cuda

DATASETS = (
    zoo.KEY_ALEXNET_IMAGENET,
    zoo.KEY_ALEXNET_PLACES365,
    zoo.KEY_RESNET152_IMAGENET,
    zoo.KEY_RESNET152_PLACES365,
    zoo.KEY_BIGGAN_IMAGENET,
    zoo.KEY_BIGGAN_PLACES365,
)

ENCODER_RESNET18 = 'resnet18'
ENCODER_RESNET50 = 'resnet50'
ENCODER_RESNET101 = 'resnet101'
ENCODERS = (ENCODER_RESNET18, ENCODER_RESNET50, ENCODER_RESNET101)

parser = argparse.ArgumentParser(description='train a captioner')
parser.add_argument(
    '--results-dir',
    type=pathlib.Path,
    help='save model to this dir (default: generated in project results dir)')
parser.add_argument('--clear-results-dir',
                    action='store_true',
                    help='clear results dir (default: do not)')
parser.add_argument('--datasets',
                    nargs='+',
                    default=DATASETS,
                    help='datasets to train on (default: all)')
parser.add_argument('--encoder',
                    choices=ENCODERS,
                    default=ENCODER_RESNET101,
                    help='image encoder (default: resnet101)')
parser.add_argument('--no-lm',
                    action='store_true',
                    help='do not train lm (default: train lm)')
parser.add_argument('--precompute-features',
                    action='store_true',
                    help='precompute image features (default: do not)')
parser.add_argument(
    '--hold-out',
    type=float,
    default=.05,
    help='hold out and validate on this fraction of training data '
    '(default: .05)')
parser.add_argument('--device', help='manually set device (default: guessed)')
args = parser.parse_args()

device = args.device or 'cuda' if cuda.is_available() else 'cpu'

results_dir: Optional[pathlib.Path] = args.results_dir
if not results_dir:
    subdir = f'captioner-{args.encoder}{"" if args.no_lm else "-lm"}'
    results_dir = env.results_dir() / subdir

if args.clear_results_dir:
    shutil.rmtree(results_dir)
results_dir.mkdir(exist_ok=True, parents=True)

dataset = zoo.datasets(*args.datasets)

splits_file = results_dir / 'splits.pth'
if splits_file.exists():
    print(f'loading cached train/test splits from {splits_file}')
    splits = torch.load(splits_file)
    train, val = training.fixed_split(dataset, splits['val'])
else:
    train, val = training.random_split(dataset, hold_out=args.hold_out)
    print(f'saving train/test splits to {splits_file}')
    torch.save({'train': train.indices, 'val': val.indices}, splits_file)

lm = None
if not args.no_lm:
    lm_file = results_dir / 'lm.pth'
    if lm_file.exists():
        print(f'loading cached lm from {lm_file}')
        lm = models.LanguageModel.load(lm_file, map_location=device)
        lm.eval()
    else:
        lm = models.lm(dataset)
        lm.fit(dataset, hold_out=val.indices, device=device)
        lm.eval()

        print(f'saving lm to {lm_file}')
        lm.save(lm_file)

encoder = models.encoder(config=args.encoder)
encoder.eval()

features = None
if args.precompute_features:
    features = encoder.map(dataset, device=device)

captioner_file = results_dir / 'captioner.pth'
if captioner_file.exists():
    print(f'loading cached decoder from {captioner_file}')
    decoder = models.Decoder.load(captioner_file, map_location=device)
    decoder.eval()
else:
    decoder = models.decoder(dataset, encoder, lm=lm)
    decoder.fit(dataset,
                features=features,
                hold_out=val.indices,
                device=device)
    decoder.eval()

    print(f'saving decoder to {captioner_file}')
    decoder.save(captioner_file)

predictions = decoder.predict(val,
                              device=device,
                              display_progress_as='caption val set')

bleu = decoder.bleu(val, predictions=predictions)
print('BLEU:', f'{bleu.score:.1f}')

bert_score = decoder.bert_score(val, predictions=predictions, device=device)
print('BERTScore:',
      ', '.join(f'{key}={val:.2f}' for key, val in bert_score.items()))
