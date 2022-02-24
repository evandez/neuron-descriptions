"""Export MILANNOTATIONS.

In some cases, we cannot release source images. This script drops the images
in those cases and packages the remaining stuff up nicely. It also handles
cases where we can release source images, e.g. with places365.
"""
import argparse
import pathlib
import re
import shutil
import tempfile

from src.utils import env

from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='zip up milannotations')
parser.add_argument('--data-dir',
                    type=pathlib.Path,
                    help='data dir (default: project data dir)')
parser.add_argument('--results-dir',
                    type=pathlib.Path,
                    help='results dir (default: project results dir)')
parser.add_argument(
    '--exclude-images',
    nargs='+',
    default=('.*(net|vgg|dino).*imagenet.*',),
    help='do not include source images when dataset matches regex')
parser.add_argument(
    '--exclude-targets',
    nargs='+',
    default=(
        r'imagenet.*',
        r'places365.*',
    ),
    help='do not package dirs matching this regex (default: imagenet, etc.)')
parser.add_argument('--targets',
                    nargs='+',
                    help='prespecified targets (default: read from data dir)')
args = parser.parse_args()

data_dir = args.data_dir or env.data_dir()

results_dir = args.results_dir or (env.results_dir() / 'export-milannotations')
results_dir.mkdir(exist_ok=True, parents=True)

targets = args.targets
if args.targets:
    targets = [data_dir / target for target in targets]
    for target in targets:
        if not target.is_dir():
            raise FileNotFoundError(f'target not found: {target}')
else:
    # Filter out non-directories.
    targets = [target for target in data_dir.iterdir() if target.is_dir()]

    # Find all subtargets, and make sure they're also subdirs.
    targets = [
        data_dir / target / subtarget
        for target in targets
        for subtarget in target.iterdir()
    ]
    targets = [target for target in targets if target.is_dir()]

    # Apply exclusions
    exclude_targets = [re.compile(exclude) for exclude in args.exclude_targets]
    targets = [
        target for target in targets if not any(
            exclude.match(str(target.relative_to(data_dir)))
            for exclude in exclude_targets)
    ]

names = '\n\t'.join(str(targ.relative_to(data_dir)) for targ in targets)
names = '\n\t' + names
print(f'found {len(targets)} export targets:{names}')

exclude_images = [re.compile(exclude) for exclude in args.exclude_images]

progress = tqdm(targets)
for target in progress:
    arch = target.parent.name
    dataset = target.name
    name = f'{arch}-{dataset}'
    progress.set_description(f'exporting {name}')
    with tempfile.TemporaryDirectory(prefix=name) as tempdir:
        temp_out_dir = pathlib.Path(tempdir)

        src_annotations_file = target / 'annotations.csv'
        if src_annotations_file.exists():
            dst_annotations_file = temp_out_dir / src_annotations_file.name
            dst_annotations_file.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(src_annotations_file, dst_annotations_file)

        # Copy layer-wise files over.
        for layer_dir in target.iterdir():
            if not layer_dir.is_dir():
                continue

            for file_name in ('masks.npy', 'ids.csv'):
                src_file = layer_dir / file_name
                if not src_file.exists():
                    raise FileNotFoundError(
                        f'missing required file: {src_file}')

                dst_file = temp_out_dir / layer_dir.name / file_name
                dst_file.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(src_file, dst_file)

            # Copy images file over if possible.
            if not any(exclude.match(dataset) for exclude in exclude_images):
                file_name = 'images.npy'
                images_src_file = layer_dir / file_name
                images_dst_file = temp_out_dir / layer_dir.name / file_name
                images_dst_file.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(images_src_file, images_dst_file)

        # Zip it up to the final output directory.
        shutil.make_archive(str(results_dir / name),
                            'zip',
                            root_dir=temp_out_dir)
