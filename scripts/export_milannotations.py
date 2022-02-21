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

from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='zip up milannotations')
parser.add_argument('root_dir', type=pathlib.Path, help='root data dir')
parser.add_argument('out_dir', type=pathlib.Path, help='output dir')
parser.add_argument(
    '--exclude-images',
    nargs='+',
    default=('imagenet*',),
    help='do not include source images when dataset matches regex')
parser.add_argument(
    '--exclude-targets',
    nargs='+',
    default=(
        r'imagenet.*',
        r'places365.*',
    ),
    help='do not package dirs matching this regex (default: imagenet, etc.)')
args = parser.parse_args()

args.out_dir.mkdir(exist_ok=True, parents=True)

# Filter out non-directories.
targets = [target for target in args.root_dir.iterdir() if target.is_dir()]

# Find all subtargets, and make sure they're also subdirs.
targets = [
    args.root_dir / target / subtarget
    for target in targets
    for subtarget in target.iterdir()
]
targets = [target for target in targets if target.is_dir()]

# Apply exclusions
exclude_targets = [re.compile(exclude) for exclude in args.exclude_targets]
targets = [
    target for target in args.root_dir.iterdir() if not any(
        exclude.match(str(target.relative_to(args.root_dir)))
        for exclude in exclude_targets)
]
print(f'found {len(targets)} export targets: {targets}')

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
        shutil.make_archive(args.out_dir / name, 'zip', root_dir=temp_out_dir)
