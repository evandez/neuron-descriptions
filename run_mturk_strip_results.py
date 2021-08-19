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
args = parser.parse_args()

hits.strip_results_csv(args.results_csv_file,
                       out_csv_file=args.out_csv_file,
                       keep_rejected=False,
                       spellcheck=True,
                       remove_prefixes=(
                           'these areas are ',
                           'these areas have ',
                           'these areas',
                           'these are ',
                           'these have ',
                           'there are ',
                           'this is ',
                           'most images contain ',
                           'most images ',
                           'the images show ',
                           'images of ',
                           'i see ',
                           'nice ',
                       ),
                       remove_substrings=(' these are ', ' nice '),
                       remove_suffixes=('.', ',', ' i can see', ' nice'))
