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
        r'imagenet',
        r'imagenet-spurious-*',
        r'places365',
    ),
    help='do not package dirs matching this regex (default: imagenet, etc.)')
args = parser.parse_args()

exclude_targets = [re.compile(exclude) for exclude in args.exclude_targets]
targets = [
    target for target in args.root_dir.iterdir()
    if target.is_dir() and not any(
        exclude.match(target.name) for exclude in exclude_targets)
]
targets = [
    args.root_dir / target / subtarget
    for target in targets
    for subtarget in target.iterdir()
]
print(f'found {len(targets)} export targets')

exclude_images = [re.compile(exclude) for exclude in args.exclude_images]

progress = tqdm(targets)
for target in progress:
    arch = target.parent.name
    dataset = target.name
    name = f'{arch}-{dataset}'
    progress.set_description(f'exporting {name}')
    with tempfile.TemporaryDirectory(prefix=name) as tempdir:
        temp_out_dir = pathlib.Path(tempdir)
        for layer in target.iterdir():
            layer_dir = target / layer

            # Copy necessary files over.
            for file_name, required in (
                ('masks.npy', True),
                ('ids.csv', True),
                ('annotations.csv', False),
            ):
                src_file = layer_dir / file_name
                assert src_file.exists, src_file

                dst_file = temp_out_dir / layer / file_name
                dst_file.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(src_file, dst_file)

            # Copy images file over if possible.
            if not any(exclude.match(dataset) for exclude in exclude_images):
                file_name = 'images.npy'
                images_src_file = layer / file_name
                images_dst_file = temp_out_dir / layer / file_name
                images_dst_file.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(images_src_file, images_dst_file)

        # Zip it up to the final output directory.
        shutil.make_archive(name,
                            'zip',
                            root_dir=args.out_dir,
                            base_dir=temp_out_dir)
