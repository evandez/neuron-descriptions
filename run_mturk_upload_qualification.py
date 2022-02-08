"""Upload the qualification to mturk."""
import argparse
import pathlib

from src.mturk import qualification

import boto3

SANDBOX_URL = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
PROD_URL = 'https://mturk-requester.us-east-1.amazonaws.com'

parser = argparse.ArgumentParser('upload qualification test')
parser.add_argument('config', type=pathlib.Path, help='path to yaml config')
parser.add_argument('--no-display-progress',
                    action='store_true',
                    help='do not show progress bar when parsing config')
parser.add_argument('--no-validate-urls',
                    action='store_true',
                    help='do not validate image urls')
parser.add_argument('--mockup-file',
                    type=pathlib.Path,
                    help='write mockup to this file')
parser.add_argument('--name',
                    default='detailed image summarizer',
                    help='qualification name (or id, if updating)')
parser.add_argument('--description',
                    help='description of the test for workers',
                    default='A simple two-question multiple choice test that '
                    'measures attention to detail in image summarization.')
parser.add_argument('--keyword',
                    action='append',
                    dest='keywords',
                    default=('image', 'summarization'),
                    help='add a keyword to the qualification')
parser.add_argument('--retry-delay-seconds',
                    type=int,
                    default=120,
                    help='retry delay in seconds')
parser.add_argument('--test-duration-seconds',
                    type=int,
                    default=1800,
                    help='maximum qualification test duration')
parser.add_argument('--aws-profile', help='aws profile to use')
parser.add_argument('--prod',
                    dest='endpoint_url',
                    action='store_const',
                    const=PROD_URL,
                    default=SANDBOX_URL,
                    help='upload to production mturk, not sandbox')
parser.add_argument('--update',
                    action='store_true',
                    help='update qualification instead of creating a new one')
args = parser.parse_args()

config = qualification.parse_yaml_config(
    args.config,
    validate_urls=not args.no_validate_urls,
    display_progress=not args.no_display_progress)
questions = qualification.generate_questions_xml(config)
answers = qualification.generate_answers_xml(config)
if args.mockup_file:
    mockup_html = qualification.generate_mockup_html(config)
    with args.mockup_file.open('w') as handle:
        handle.write(mockup_html)

boto3.setup_default_session(profile_name=args.aws_profile)
client = boto3.client('mturk', endpoint_url=args.endpoint_url)
if args.update:
    client.update_qualification_type(
        QualificationTypeId=args.name,
        RetryDelayInSeconds=args.retry_delay_seconds,
        QualificationTypeStatus='Active',
        Description=args.description,
        Test=questions,
        AnswerKey=answers,
        TestDurationInSeconds=args.test_duration_seconds,
        AutoGranted=False,
    )
else:
    client.create_qualification_type(
        Name=args.name,
        Keywords=' '.join(args.keywords),
        Description=args.description,
        QualificationTypeStatus='Active',
        RetryDelayInSeconds=args.retry_delay_seconds,
        Test=questions,
        AnswerKey=answers,
        TestDurationInSeconds=args.test_duration_seconds,
        AutoGranted=False,
    )
