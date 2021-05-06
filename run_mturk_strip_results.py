"""Strip MTurk results CSV of unnecessary info so it can be distributed."""
import argparse
import pathlib

from lv.mturk import hits

parser = argparse.ArgumentParser(description='strip mturk results csv')
parser.add_argument('results_csv_file', type=pathlib.Path, help='results csv')
parser.add_argument(
    '--out-csv-file',
    type=pathlib.Path,
    help='write stripped results here; by default, overwrites input file')
args = parser.parse_args()

hits.strip_results_csv(args.results_csv_file, out_csv_file=args.out_csv_file)
