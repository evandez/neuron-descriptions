"""Strip MTurk results CSV of unnecessary info so it can be distributed."""
import argparse
import pathlib

from src.mturk import hits

parser = argparse.ArgumentParser(description='strip mturk results csv')
parser.add_argument('results_csv_file', type=pathlib.Path, help='results csv')
parser.add_argument(
    '--out-csv-file',
    type=pathlib.Path,
    help='write stripped results here; by default, overwrites input file '
    '(default: overwrite original)')
parser.add_argument('--replace-space-around-for',
                    help='replace all "space around" phrases for this worker '
                    '(default: none)')
parser.add_argument('--replace-for-worker',
                    dest='replacements_by_worker',
                    nargs=3,
                    action='append',
                    help='replace substrings for this worker (default: none)')
parser.add_argument('--legacy',
                    action='store_true',
                    help='if set, parse layer/unit from image url '
                    '(default: use layer/unit columns)')
args = parser.parse_args()

replacements_by_worker = list(args.replacements_by_worker or [])

# One worker prefixed thousands of annotations with phrases like
# "space around", making this phrase uninformative. Hence, we remove all
# phrases like this from their annotations specifically. The ID must be passed
# as an argument for privacy reasons.
replace_space_around_for = args.replace_space_around_for
if replace_space_around_for:
    for noun in ('space', 'spaces'):
        for preposition in ('around', 'along', 'to', 'in'):
            for article in ('a ', 'an ', 'the ', ''):
                phrase = f'{noun} {preposition} {article}'
                replacements_by_worker.append(
                    (replace_space_around_for, phrase, ''))
            for punct in ('.', ','):
                phrase = f' {noun} {preposition}{punct}'
                replacements_by_worker.append(
                    (replace_space_around_for, phrase, ''))

    # Some other oddball phrases...
    for phrase in ('with space', 'with space around'):
        for punct in (',', '.'):
            replacements_by_worker.append(
                (replacements_by_worker, f' {phrase}{punct}', ''))


def replace_worker_specific(annotation: str, row: hits.ResultsRow) -> str:
    """Make a worker-specific transformation to the annotation."""
    for worker_id, old_str, new_str in replacements_by_worker:
        if row['WorkerId'] == worker_id:
            annotation = annotation.replace(old_str, new_str)
    return annotation


results_csv_file = args.results_csv_file
out_csv_file = args.out_csv_file
legacy = args.legacy

hits.strip_results_csv(
    results_csv_file,
    out_csv_file=out_csv_file,
    in_layer_column='Input.image_url_1' if args.legacy else 'Input.layer',
    in_unit_column='Input.image_url_1' if args.legacy else 'Input.unit',
    transform_layer=(lambda url, _: url.split('/')[-5]) if legacy else None,
    transform_unit=(lambda url, _: url.split('/')[-2][5:]) if legacy else None,
    transform_annotation=replace_worker_specific,
    keep_rejected=False,
    spellcheck=True,
    remove_prefixes=(
        'a bunch of ',
        'a lot of different kinds of ',
        'all are ',
        'all have ',
        'all images are ',
        'all images include ',
        'all images contain ',
        'all the above are ',
        'all ',
        'areas of ',
        'area of ',
        'different kinds of ',
        'different types of ',
        'different ',
        'images of ',
        'images show ',
        'images have ',
        'images with ',
        'in different ',
        'it is the ',
        'it is ',
        'it shows an image that ',
        'it shows an image ',
        'it shows ',
        'i see ',
        'many different types of ',
        'many different kinds of ',
        'many different ',
        'most images contain ',
        'most of the images ',
        'most images ',
        'most ',
        'nice ',
        'of these ',
        'these areas are ',
        'these areas have ',
        'these areas ',
        'these area ',
        'these items are ',
        'these items ',
        'these regions have ',
        'these regions show ',
        'these regions are ',
        'these regions ',
        'these pictures all have ',
        'these pictures all show ',
        'these pictures are ',
        'these pictures show ',
        'these pictures have ',
        'these pictures ',
        'these are ',
        'these is ',
        'these have ',
        'these show ',
        'these contain ',
        'these look like ',
        'there mainly are ',
        'there is ',
        'there are ',
        'they are ',
        'they all are ',
        'they have ',
        'they all have ',
        'they all contain ',
        'they all show ',
        'they create ',
        'they ',
        'this is ',
        'this looks like ',
        'this collection depicts ',
        'this collection ',
        'the images presented show ',
        'the images show ',
        'the image of ',
        'these images have ',
        'the regions have ',
        'the is the ',
        'the is ',
    ),
    remove_suffixes=(
        '.',
        ',',
        ' i can see',
        ' nice',
        ', is shown',
        ', are shown',
        ' is shown',
        ' are shown',
        ', space around',

        # These fixes are a bit controversial, as they *could* refer to a
        # positional relationship--but it looks like most of the time, it's
        # the suffix to "The regions have * in them."
        ' in them',
        ' within them',
    ),
    remove_substrings=(' we can see ',),
    replace_substrings={
        # Words that are commonly accidentally joined...
        'andflower': 'and flower',
        'andvehicles': 'and vehicles',
        'andwhite': 'and white',
        'archbridge': 'arch bridge',
        'archwindow': 'arch window',
        'aroundanimal': 'around animal',
        'aroundclothing': 'around clothing',
        'bodypart': 'body part',
        'bottlecaps': 'bottle caps',
        'bridgepathway': 'bridge pathway',
        'collarbelt': 'collar belt',
        'crosshatching': 'cross hatching',
        'dirtbike': 'dirt bike',
        'dunebuggies': 'dune buggies',
        'fenceposts': 'fence posts',
        'fireescape': 'fire escape',
        'fireexit': 'fire exit',
        'fourposter': 'four poster',
        'gaspump': 'gas pump',
        'golfcart': 'golf cart',
        'glasswindshield': 'glass windshield',
        'grassplain': 'grass plain',
        'groundway': 'ground way',
        'haybale': 'hay bale',
        'hockeyplayer': 'hockey player',
        'housefront': 'house front',
        'jackolantern': 'jack o lantern',
        'jack o\' lantern': 'jack o lantern',
        'neckcollar': 'neck collar',
        'largebuilding': 'large building',
        'licenseplate': 'license plate',
        'lightpole': 'light pole',
        'lightswitch': 'light switch',
        'lockerroom': 'locker room',
        'multitexture': 'multi texture',
        'ofdistorted': 'of distorted',
        'ofknitted': 'of knitted',
        'ofsimilar': 'of similar',
        'onetower': 'one tower',
        'peoplewalking': 'people walking',
        'plantlife': 'plant life',
        'rockcliff': 'rock cliff',
        'rockformation': 'rock formation',
        'showercap': 'shower cap',
        'spacearound': 'space around',
        'spacesaround': 'spaces around',
        'spacebelow': 'space below',
        'spacebetween': 'space between',
        'sportcar': 'sport car',
        'starfish': 'star fish',
        'sticklike': 'stick like',
        'stonebuilding': 'stone building',
        'stonebuiding': 'stone building',
        'stonepath': 'stone path',
        'streetcorner': 'street corner',
        'subwaycar': 'subway car',
        'telephonebox': 'telephone box',
        'theback': 'the back',
        'thebackground': 'the background',
        'thecarpet': 'the carpet',
        'theclothing': 'the clothing',
        'thedistance': 'the distance',
        'thefeather': 'the feather',
        'thegravel': 'the gravel',
        'thepavement': 'the pavement',
        'thesethese': 'these',
        'thesky': 'the sky',
        'thesticker': 'the sticker',
        'theswimming': 'the swimming',
        'theletter': 'the letter',
        'thewindow': 'the window',
        'trainstop': 'train stop',
        'traintrack': 'train track',
        'trainyard': 'train yard',
        'treebranch': 'tree branch',
        'treefront': 'tree front',
        'treesnear': 'trees near',
        'totempole': 'totem pole',
        'watersource': 'water source',
        'waterfront': 'water front',
        'waterbottle': 'water bottle',
        'watertowers': 'water towers',
        'webpage': 'web page',
        "''": "'",

        # Typos not caught by spellcheck.
        ' og ': ' of ',
        'aanndanimals': 'and animals',
        'aorunditem': 'around item',
        'aqauticlife': 'aquatic life',
        'bridgewalkay': 'bridge walkway',
        'camoflouged': 'camoflauged',
        'ciggerate': 'cigarette',
        'limehouse': 'lighthouse',
        'passangercar': 'passenger car',
        'rockclif': 'rock cliff',
        'showbtwo': 'show two',
        'spacearoudn': 'space around',
        'spacearoun': 'space around',
        'treeeeeeeeee': 'tree',
        'watertown': 'water tower',
        'thedessort': 'the desert',
        'uliitity': 'utility',
        'building.space': 'building. space',
        'designs.these': 'designs. these',
        'designs.regions': 'designs. regions',
        'food.space': 'food. space',
        ',sappce': ', space',

        # Other boilerplate phrases/mistakes that need to be carefully fixed.
        ' these are ': ' ',
        ' nice ': ' ',
        ' i ': ' ',

        # Some very specific mistakes...
        'theface ': 'the face ',
        ' asign': ' a sign',
        ' dres,': ' dress,',
        ' ona ': ' on a ',
        ' ofa ': ' of a ',
        'tree branch, space around a tree branch': 'tree branch',
        'arounda': 'around a',
    },
    replace_exact={
        'none of the above': 'nothing',
    })
