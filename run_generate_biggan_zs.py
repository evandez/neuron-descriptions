"""Generate zs (and/or ys) for pretrained GANs."""
import argparse
import pathlib

from src.deps.pretorched.gans import biggan, utils

import torch

parser = argparse.ArgumentParser(description='generate a bunch of gan inputs')
parser.add_argument('dataset',
                    choices=('imagenet', 'places365'),
                    help='dataset model was trained on')
parser.add_argument('path', type=pathlib.Path, help='write zs and ys here')
parser.add_argument('--num-samples',
                    '-n',
                    dest='n',
                    type=int,
                    default=100000,
                    help='number of samples to generate (default: 100k)')
args = parser.parse_args()

model = biggan.BigGAN(pretrained=args.dataset, device='cpu')
n_classes = 1000 if args.dataset == 'imagenet' else 365
zs, _ = utils.prepare_z_y(args.n, model.dim_z, n_classes, device='cpu')
ys = torch.randint(n_classes, size=(args.n,))

args.path.parent.mkdir(exist_ok=True, parents=True)
torch.save((zs, ys), args.path)
