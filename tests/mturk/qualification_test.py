"""Tests for the `src.mturk.qualification` module."""
import pathlib
import tempfile
from xml.etree import ElementTree

from src.mturk import qualification

import pytest

TITLE = 'My Qualification Test'
INSTRUCTIONS = 'Please answer all of the questions'

QUESTION_1_ID = 'q1'
QUESTION_2_ID = 'q2'

IMAGE_URL_1 = 'https://images.com/my_image_1.png'
IMAGE_URL_2 = 'https://images.com/my_image_2.png'
IMAGE_URL_3 = 'https://images.com/my_image_3.png'
IMAGE_URL_4 = 'https://images.com/my_image_4.png'

ANSWER_OPTION_1 = 'answer option 1'
ANSWER_OPTION_2 = 'answer option 2'
ANSWER_OPTION_3 = 'answer option 3'
ANSWER_OPTION_4 = 'answer option 4'

QUESTION_1_OPTIONS = (ANSWER_OPTION_1, ANSWER_OPTION_2)
QUESTION_2_OPTIONS = (ANSWER_OPTION_3, ANSWER_OPTION_4)

QUESTION_1_ANSWER = 0
QUESTION_2_ANSWER = 1


@pytest.fixture
def config():
    """Return Config for testing."""
    return qualification.Config(
        title=TITLE,
        instructions=INSTRUCTIONS,
        questions=(
            qualification.Question(
                question_id=QUESTION_1_ID,
                image_urls=(IMAGE_URL_1, IMAGE_URL_2),
                options=QUESTION_1_OPTIONS,
                answer_index=QUESTION_1_ANSWER,
            ),
            qualification.Question(
                question_id=QUESTION_2_ID,
                image_urls=(IMAGE_URL_3, IMAGE_URL_4),
                options=QUESTION_2_OPTIONS,
                answer_index=QUESTION_2_ANSWER,
            ),
        ),
    )


YAML_CONFIG = f'''\
title: {TITLE}
instructions: {INSTRUCTIONS}
questions:
    - id: {QUESTION_1_ID}
      image_urls:
          - {IMAGE_URL_1}
          - {IMAGE_URL_2}
      options:
          - {ANSWER_OPTION_1}
          - {ANSWER_OPTION_2}
      answer_index: {QUESTION_1_ANSWER}
    - id: {QUESTION_2_ID}
      image_urls:
          - {IMAGE_URL_3}
          - {IMAGE_URL_4}
      options:
          - {ANSWER_OPTION_3}
          - {ANSWER_OPTION_4}
      answer_index: {QUESTION_2_ANSWER}
'''


@pytest.yield_fixture
def yaml_file():
    """Yield a YAML config file for testing."""
    with tempfile.TemporaryDirectory() as tempdir:
        file = pathlib.Path(tempdir) / 'config.yaml'
        with file.open('w') as handle:
            handle.write(YAML_CONFIG)
        yield file


def test_parse_yaml_config(yaml_file, config):
    """Test parse_yaml_config correctly reads file."""
    actual = qualification.parse_yaml_config(yaml_file, validate_urls=False)
    print(actual, config)
    assert actual == config


@pytest.mark.parametrize(
    'yaml_config,error_pattern',
    (
        ('foo: bar', '.*title.*'),
        ('title: foo', '.*instructions.*'),
        ('title: foo\ninstructions: bar', '.*>= 1 questions.*'),
        (
            '''\
title: foo
instructions: bar
questions:
    - image_urls:
        - foo
      options:
        - foo
      answer_index: 1''',
            '.*question "id".*',
        ),
        (
            '''\
title: foo
instructions: bar
questions:
    - id: foo
      options:
        - foo
      answer_index: 1''',
            '.*image_urls.*',
        ),
        (
            '''\
title: foo
instructions: bar
questions:
    - id: foo
      image_urls:
      options:
        - foo
      answer_index: 1''',
            '.*image_urls.*',
        ),
        (
            '''\
title: foo
instructions: bar
questions:
    - id: foo
      image_urls:
        - url.com
      answer_index: 1''',
            '.*options.*',
        ),
        (
            '''\
title: foo
instructions: bar
questions:
    - id: foo
      image_urls:
        - url.com
      options:
      answer_index: 1''',
            '.*options.*',
        ),
        (
            '''\
title: foo
instructions: bar
questions:
    - id: foo
      image_urls:
        - url.com
      options:
        - option 1
        - option 2''',
            '.*answer_index.*',
        ),
    ),
)
def test_parse_yaml_config_bad_config(yaml_file, yaml_config, error_pattern):
    """Test parse_yaml_config dies when given bad config."""
    with yaml_file.open('w') as handle:
        handle.write(yaml_config)
    with pytest.raises(ValueError, match=error_pattern):
        qualification.parse_yaml_config(yaml_file,
                                        validate_urls=False,
                                        display_progress=False)


QUESTIONS_XML = f'''\
<QuestionForm xmlns="{qualification.QUESTION_FORM_XMLNS}">
<Overview>
<Title>
{TITLE}
</Title>
<Text>
{INSTRUCTIONS}
</Text>
</Overview>
<Question>
<QuestionIdentifier>
{QUESTION_1_ID}
</QuestionIdentifier>
<DisplayName>
Question 1
</DisplayName>
<IsRequired>true</IsRequired>
<QuestionContent>
<FormattedContent><![CDATA[
<table><tr>
<td><img src="{IMAGE_URL_1}" alt="image 1"/></td>
<td><img src="{IMAGE_URL_2}" alt="image 2"/></td>
</tr></table>
]]></FormattedContent>
</QuestionContent>
<AnswerSpecification>
<SelectionAnswer>
<Selections>
<Selection>
<SelectionIdentifier>
{ANSWER_OPTION_1.replace(" ", "_")}
</SelectionIdentifier>
<Text>
{ANSWER_OPTION_1}
</Text>
</Selection>
<Selection>
<SelectionIdentifier>
{ANSWER_OPTION_2.replace(" ", "_")}
</SelectionIdentifier>
<Text>
{ANSWER_OPTION_2}
</Text>
</Selection>
</Selections>
</SelectionAnswer>
</AnswerSpecification>
</Question>
<Question>
<QuestionIdentifier>
{QUESTION_2_ID}
</QuestionIdentifier>
<DisplayName>
Question 2
</DisplayName>
<IsRequired>true</IsRequired>
<QuestionContent>
<FormattedContent><![CDATA[
<table><tr>
<td><img src="{IMAGE_URL_3}" alt="image 1"/></td>
<td><img src="{IMAGE_URL_4}" alt="image 2"/></td>
</tr></table>
]]></FormattedContent>
</QuestionContent>
<AnswerSpecification>
<SelectionAnswer>
<Selections>
<Selection>
<SelectionIdentifier>
{ANSWER_OPTION_3.replace(" ", "_")}
</SelectionIdentifier>
<Text>
{ANSWER_OPTION_3}
</Text>
</Selection>
<Selection>
<SelectionIdentifier>
{ANSWER_OPTION_4.replace(" ", "_")}
</SelectionIdentifier>
<Text>
{ANSWER_OPTION_4}
</Text>
</Selection>
</Selections>
</SelectionAnswer>
</AnswerSpecification>
</Question>
</QuestionForm>
'''


def test_generate_questions_xml(config):
    """Test generate_questions_xml returns expected questions XML."""
    actual = qualification.generate_questions_xml(config)
    actual = ElementTree.tostring(ElementTree.fromstring(actual))
    expected = ElementTree.tostring(ElementTree.fromstring(QUESTIONS_XML))
    assert actual == expected


ANSWERS_XML = f'''\
<AnswerKey xmlns="{qualification.ANSWER_KEY_XMLNS}">
<Question>
<QuestionIdentifier>
{QUESTION_1_ID}
</QuestionIdentifier>
<AnswerOption>
<SelectionIdentifier>
{QUESTION_1_OPTIONS[QUESTION_1_ANSWER].replace(' ', '_')}
</SelectionIdentifier>
<AnswerScore>1</AnswerScore>
</AnswerOption>
</Question>
<Question>
<QuestionIdentifier>
{QUESTION_2_ID}
</QuestionIdentifier>
<AnswerOption>
<SelectionIdentifier>
{QUESTION_2_OPTIONS[QUESTION_2_ANSWER].replace(' ', '_')}
</SelectionIdentifier>
<AnswerScore>1</AnswerScore>
</AnswerOption>
</Question>
<QualificationValueMapping>
<PercentageMapping>
<MaximumSummedScore>2</MaximumSummedScore>
</PercentageMapping>
</QualificationValueMapping>
</AnswerKey>
'''


def test_generate_answers_xml(config):
    """Test generate_answers_xml returns expected answers XML."""
    actual = qualification.generate_answers_xml(config)
    actual = ElementTree.tostring(ElementTree.fromstring(actual))
    expected = ElementTree.tostring(ElementTree.fromstring(ANSWERS_XML))
    assert actual == expected


MOCKUP_HTML = f'''\
<!DOCTYPE html>
<html>
<body>
<h2>{TITLE}</h2>
<p>{INSTRUCTIONS}</p>
<h3>Question 1</h3>
<table>
<tr>
<td><img src="{IMAGE_URL_1}"/></td>
<td><img src="{IMAGE_URL_2}"/></td>
</tr>
</table>
<ol type="A">
<li>
{ANSWER_OPTION_1} (id {ANSWER_OPTION_1.replace(' ', '_')})
</li>
<li>
{ANSWER_OPTION_2} (id {ANSWER_OPTION_2.replace(' ', '_')})
</li>
</ol>
<h3>Question 2</h3>
<table>
<tr>
<td><img src="{IMAGE_URL_3}"/></td>
<td><img src="{IMAGE_URL_4}"/></td>
</tr>
</table>
<ol type="A">
<li>
{ANSWER_OPTION_3} (id {ANSWER_OPTION_3.replace(' ', '_')})
</li>
<li>
{ANSWER_OPTION_4} (id {ANSWER_OPTION_4.replace(' ', '_')})
</li>
</ol>
</body>
</html>
'''


def test_generate_mockup_html(config):
    """Test generate_mockup_html returns expected html."""
    actual = qualification.generate_mockup_html(config)
    actual = ElementTree.tostring(ElementTree.fromstring(actual))
    expected = ElementTree.tostring(ElementTree.fromstring(MOCKUP_HTML))
    assert actual == expected
