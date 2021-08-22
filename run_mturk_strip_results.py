"""Strip MTurk results CSV of unnecessary info so it can be distributed."""
import argparse
import pathlib

from lv.mturk import hits

parser = argparse.ArgumentParser(description='strip mturk results csv')
parser.add_argument('results_csv_file', type=pathlib.Path, help='results csv')
parser.add_argument(
    '--out-csv-file',
    type=pathlib.Path,
    help='write stripped results here; by default, overwrites input file '
    '(default: overwrite original)')
parser.add_argument('--legacy',
                    action='store_true',
                    help='if set, parse layer/unit from image url '
                    '(default: use layer/unit columns)')
args = parser.parse_args()

results_csv_file = args.results_csv_file
out_csv_file = args.out_csv_file
legacy = args.legacy

hits.strip_results_csv(
    results_csv_file,
    out_csv_file=out_csv_file,
    in_layer_column='Input.image_url_1' if args.legacy else 'Input.layer',
    in_unit_column='Input.image_url_1' if args.legacy else 'Input.unit',
    transform_layer=(lambda url: url.split('/')[-5]) if legacy else None,
    transform_unit=(lambda url: url.split('/')[-2][5:]) if legacy else None,
    keep_rejected=False,
    spellcheck=True,
    remove_prefixes=(
        'these areas are ',
        'these areas have ',
        'these areas',
        'these are ',
        'these have ',
        'there are ',
        'they are ',
        'they all are ',
        'this is ',
        'most images contain ',
        'all are ',
        'the is the ',
        'all images are ',
        'all images include ',
        'all images contain ',
        'all the above are ',
        'most images ',
        'the images show ',
        'images of ',
        'i see ',
        'nice ',
    ),
    remove_substrings=(' these are ', ' nice '),
    remove_suffixes=(
        '.',
        ',',
        ' i can see',
        ' nice',
        ', is shown',
        ', are shown',
        ' is shown',
        ' are shown',
    ),
    replace_substrings={
        'none of the above': 'nothing',
        ' og ': ' of ',
    })
