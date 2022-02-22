"""Little script for exporting BigGAN zs/ys used in data collection."""
import argparse
import pathlib
import shutil

from src.utils import env

from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='export biggan zs')
parser.add_argument('--data-dir',
                    type=pathlib.Path,
                    help='root data dir (default: project data dir)')
parser.add_argument('--results-dir',
                    type=pathlib.Path,
                    help='results dir (default: project results dir)')
parser.add_argument('--datasets',
                    nargs='+',
                    default=('imagenet', 'places365'),
                    help='versions of biggan to export zs for (default: all)')
args = parser.parse_args()

data_dir = args.data_dir or env.data_dir()

results_dir = args.results_dir or (env.results_dir() / 'export-biggan-zs')
results_dir.mkdir(exist_ok=True, parents=True)

for dataset in tqdm(args.datasets):
    zs_dir = data_dir / f'biggan-zs-{dataset}'
    if not zs_dir.is_dir():
        raise FileNotFoundError(f'zs dataset not found: {zs_dir}')

    shutil.make_archive(str(results_dir / zs_dir.name),
                        'zip',
                        root_dir=data_dir,
                        base_dir=zs_dir.relative_to(data_dir))
