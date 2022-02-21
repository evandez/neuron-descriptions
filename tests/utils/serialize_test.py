"""Unit tests for `src.utils.serialize` module."""
import dataclasses
import pathlib
import tempfile

from src.utils import serialize

import pytest
import spacy
from spacy.lang import en
from torch import nn

STRING = 'my_str'
NUMBER = 123


@dataclasses.dataclass(frozen=True)
class SerializableObject(serialize.Serializable):
    """A simple object that can be serialized."""

    string: str
    number: int


@pytest.fixture
def serializable_object():
    """Return a SerializableObject for testing."""
    return SerializableObject(STRING, NUMBER)


def test_serializable_properties(serializable_object):
    """Test Serializable.properties returns all fields by default."""
    actual = serializable_object.properties()
    assert actual == {'string': STRING, 'number': NUMBER}


def test_serializable_serializable(serializable_object):
    """Test Serializable.serializable returns nothing by default."""
    actual = serializable_object.serializable()
    assert actual == {}


def test_serializble_serialize(serializable_object):
    """Test Serializable.serialize correctly serializes object."""
    actual = serializable_object.serialize()
    assert actual == {
        'properties': {
            'string': STRING,
            'number': NUMBER
        },
        'children': {}
    }


def test_serializable_deserialize():
    """Test Serializable.deserialize correctly deserializes object."""
    serialized = {
        'properties': {
            'string': STRING,
            'number': NUMBER
        },
        'children': {}
    }
    actual = SerializableObject.deserialize(serialized)
    assert isinstance(actual, SerializableObject)
    assert actual.string == STRING
    assert actual.number == NUMBER


def test_serializable_resolve():
    """Test `Serializable.resolve` does nothing by default."""
    actual = SerializableObject.resolve({})
    assert actual == {}


@dataclasses.dataclass(frozen=True)
class SerializableObjectWithNLP(SerializableObject):
    """A more complicated object with an NLP field."""

    nlp: en.English


@pytest.fixture(scope='module')
def nlp():
    """Return a singleton fake NLP object for testing."""
    return spacy.load('en_core_web_sm')


@pytest.fixture
def serializable_object_with_nlp(nlp):
    """Return a SerializableObjectWithNLP for testing."""
    return SerializableObjectWithNLP(STRING, NUMBER, nlp)


def test_serializable_serialize_nlp(serializable_object_with_nlp):
    """Test Serializable.serialize handles spacy objects."""
    actual = serializable_object_with_nlp.serialize()

    assert actual.keys() == {'properties', 'children'}
    assert actual['children'] == {}

    properties = actual['properties']
    assert properties.keys() == {'string', 'number', 'nlp'}
    assert properties['string'] == STRING
    assert properties['number'] == NUMBER
    assert isinstance(properties['nlp'], tuple)
    assert len(properties['nlp']) == 2


TEXT = 'here is a "sentence".'


def test_serializable_deserialize_nlp(serializable_object_with_nlp):
    """Test Serializable.deserialize handles spacy objects."""
    serialized = serializable_object_with_nlp.serialize()
    actual = SerializableObjectWithNLP.deserialize(serialized)
    assert actual.string == STRING
    assert actual.number == NUMBER

    before_nlp = serializable_object_with_nlp.nlp
    after_nlp = actual.nlp
    before_tokens = [tok.text for tok in before_nlp(TEXT)]
    after_tokens = [tok.text for tok in after_nlp(TEXT)]
    assert before_tokens == after_tokens


CHILD_KEY = 'child'
CHILD_TYPE_KEY = 'serializable_object'


@dataclasses.dataclass(frozen=True)
class RecursivelySerializableObject(SerializableObject):
    """A serializable object with recursively serializable fields."""

    child: SerializableObject

    def serializable(self):
        """Return serializable property map."""
        return {CHILD_KEY: CHILD_TYPE_KEY}

    @classmethod
    def resolve(self, children):
        """Resolve types of child serializable properties."""
        assert children == {CHILD_KEY: CHILD_TYPE_KEY}
        return {CHILD_KEY: SerializableObject}


CHILD_STRING = 'child-string'
CHILD_NUMBER = 456


@pytest.fixture
def recursively_serializable_object():
    """Return a RecursivelySerializableType for testing."""
    return RecursivelySerializableObject(
        STRING, NUMBER, SerializableObject(CHILD_STRING, CHILD_NUMBER))


def test_serializable_serialize_recursive(recursively_serializable_object):
    """Test Serializable.serialize handles recursive serialization."""
    actual = recursively_serializable_object.serialize()
    assert actual == {
        'properties': {
            'string': STRING,
            'number': NUMBER,
            CHILD_KEY: {
                'properties': {
                    'string': CHILD_STRING,
                    'number': CHILD_NUMBER,
                },
                'children': {},
            },
        },
        'children': {
            CHILD_KEY: CHILD_TYPE_KEY,
        },
    }


def test_serializable_deserialize_recursive(recursively_serializable_object):
    """Test Serializable.deserialize handles recursive deserialization."""
    serialized = recursively_serializable_object.serialize()
    actual = recursively_serializable_object.deserialize(serialized)
    assert actual.string == STRING
    assert actual.number == NUMBER
    assert actual.child.string == CHILD_STRING
    assert actual.child.number == CHILD_NUMBER


IN_FEATURES = 5
OUT_FEATURES = 10


class SerializableModule(serialize.SerializableModule):
    """A simple serializable module."""

    def __init__(self, in_features, out_features):
        """Initialize the module."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def properties(self, **_):
        """Return serializable properties for the module."""
        return {
            'in_features': self.in_features,
            'out_features': self.out_features,
        }


@pytest.fixture
def serializable_module():
    """Return a SerializableModule for testing."""
    return SerializableModule(IN_FEATURES, OUT_FEATURES)


@pytest.mark.parametrize('state_dict', (False, True))
def test_serializable_module_serialize(serializable_module, state_dict):
    """Test SerializableModule.serialize correctly serializes module."""
    actual = serializable_module.serialize(state_dict=state_dict)

    assert 'properties' in actual
    assert actual['properties'] == {
        'in_features': IN_FEATURES,
        'out_features': OUT_FEATURES,
    }

    assert 'children' in actual
    assert actual['children'] == {}

    if state_dict:
        assert 'state_dict' in actual
        assert len(actual['state_dict']) == 2


def test_serializable_module_deserialize(serializable_module):
    """Test SerializableModule.deserialize correctly recovers object."""
    serialized = serializable_module.serialize()
    actual = SerializableModule.deserialize(serialized)
    assert actual.in_features == IN_FEATURES
    assert actual.out_features == OUT_FEATURES
    for aparam, eparam in zip(actual.parameters(),
                              serializable_module.parameters()):
        assert aparam.allclose(eparam)


def test_serializable_module_deserialize_no_state_dict(serializable_module):
    """Test SerializableModule.deserialize supports not loading params."""
    serialized = serializable_module.serialize()
    actual = SerializableModule.deserialize(serialized, load_state_dict=False)
    assert actual.in_features == IN_FEATURES
    assert actual.out_features == OUT_FEATURES
    assert not all(
        aparam.allclose(eparam) for aparam, eparam in zip(
            actual.parameters(), serializable_module.parameters()))


def test_serializable_module_save_load(serializable_module):
    """Test SerializableModule.save and SerializableModule.load cooperate."""
    with tempfile.TemporaryDirectory() as tempdir:
        file = pathlib.Path(tempdir) / 'model.pth'
        serializable_module.save(file)
        actual = SerializableModule.load(file)
    assert actual.in_features == IN_FEATURES
    assert actual.out_features == OUT_FEATURES
    for aparam, eparam in zip(actual.parameters(),
                              serializable_module.parameters()):
        assert aparam.allclose(eparam)
