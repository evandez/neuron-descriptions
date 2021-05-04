"""Utilities for serializing arbitrary objects."""
from typing import Any, Mapping, Type, TypeVar

from lv.utils.typing import PathLike

import spacy
import torch
from torch import nn

SerializableT = TypeVar('SerializableT', bound='Serializable')


class Serializable:
    """Mixin that makes an object (loosely) serializable.

    Here, serialization means mapping an object to an equivalent dictionary.
    The object should be reconstructable from that dictionary alone, but the
    serialization procedure need not be the same as doing `vars(obj)` even
    though that is the default behavior for this mixin.

    The object will be deserialized by passing the entire dictionary as keyword
    arguments to the constructor. Subclasses should override the `properties`
    method so that the properties exactly reflect the kwargs required by the
    constructor.

    This mixin does recursively serialize some common objects, most notably
    spacy `Language` objects. It also automatically serializes any field that
    inherits from `Serializable`. HOWEVER, it does *not* automatically
    deserialize other Serializable objects. For recursive deserialization, you
    must specify the child types by overriding `Serializable.recurse`. By doing
    it this way, we can avoid saving the type information with the payload,
    supporting better cross-codebase transfer of the data.

    If you don't like this, tough luck. Go use pickle, why don't ya? The use
    case for this mixin is to get the general serialization behavior of pickle
    without tying the payloads to this codebase: you can open the payloads
    anywhere, as they're just dictionaries, and inspect the data yourself or
    even deserialize the object yourself. This is especially useful for
    distributing research models, which will be used across many different
    codebases in settings where the source is heavily, frequently, and
    indiscriminately modified.
    """

    def __init__(self, **_):
        """Initialize the object (necessary for type checking)."""
        super().__init__()

    def properties(self, **_) -> Mapping[str, Any]:
        """Return all properties needed to reconstruct the object.

        Properties are basically constructor (kw)args. Keyword arguments are
        arbitrary options for configuring how object is serialized.

        By default, returns dictionary of object fields. Subclasses should
        override this method if different behavior is desired.
        """
        return vars(self)

    def serialize(self, **kwargs: Any) -> Mapping[str, Any]:
        """Return object serialized as a dictionary.

        Keyword arguments are forwarded to `properties`.
        """
        properties = dict(self.properties(**kwargs))

        # Some object types require special serialization. We'll search for
        # those objects and do it live.
        queue = [properties]
        while queue:
            current = queue.pop()
            for key, value in current.items():
                if isinstance(value, dict):
                    queue.append(value)
                elif isinstance(value, spacy.Language):
                    config = value.config
                    payload = value.to_bytes()
                    current[key] = (config, payload)

        # Handle recursive serialization.
        recurse = self.recurse()
        for key, value in properties.items():
            if key in recurse:
                if not isinstance(value, Serializable):
                    raise ValueError('cannot recurse on non-serializable'
                                     f'type: {type(value).__name__}')
                properties[key] = value.serialize()

        return properties

    @classmethod
    def deserialize(cls: Type[SerializableT],
                    properties: Mapping[str, Any],
                    strict: bool = False) -> SerializableT:
        """Deserialize the object from its properties.

        Args:
            properties (Mapping[str, Any]): The object properties.
            strict (bool, optional): If set, die when a recursive deserialize
                is specified but not performed. Defaults to False.

        Returns:
            SerializableT: The deserialized object.

        """
        properties = dict(properties)

        # First deserialize some known properties.
        queue = [properties]
        while queue:
            current = queue.pop()
            for key, value in current.items():
                if isinstance(value, dict):
                    queue.append(value)
                elif isinstance(value, tuple) and len(value) == 2:
                    config, payload = value
                    if isinstance(config, dict) and isinstance(payload, bytes):
                        lang = spacy.util.get_lang_class(config['nlp']['lang'])
                        nlp = lang.from_config(config)
                        nlp.from_bytes(payload)
                        current[key] = nlp

        # Then handle recursion, if requested and if possible.
        recurse = cls.recurse()
        for key, SerializableType in recurse.items():
            if strict and key not in properties:
                raise KeyError(f'cannot recurse on {key}; not in properties')
            properties[key] = SerializableType.deserialize(properties[key])

        deserialized = cls(**properties)
        return deserialized

    @classmethod
    def recurse(cls) -> Mapping[str, Type['Serializable']]:
        """Return all recursively serializable types for this class."""
        return {}


SerializableModuleT = TypeVar('SerializableModuleT',
                              bound='SerializableModule')


class SerializableModule(Serializable, nn.Module):
    """A serializable torch module.

    This class provides `save` and `load` functions in addition to the
    `serialize` and `deserialize` functions provided by `Serializable`.
    Additionally, if one of the properties is named `state_dict`, this class
    will call `load_state_dict` on it.
    """

    def __init__(self, **_: Any):
        """Initialize the module."""
        super().__init__()

    def properties(self,
                   state_dict: bool = True,
                   **kwargs) -> Mapping[str, Any]:
        """Return default serializable values (i.e. module parameters).

        Keyword arguments are forwarded to `torch.nn.Module.state_dict`.

        Args:
            state_dict (bool, optional): If set, incude state dict as a
                property. Defaults to True.

        Returns:
            Mapping[str, Any]: Empty dict if `state_dict=False`, otherwise
                a dictionary of the form `{'state_dict': ...}`.

        """
        properties = {}
        if state_dict:
            properties['state_dict'] = self.state_dict(**kwargs)
        return properties

    def save(self, file: PathLike, **kwargs: Any) -> None:
        """Save the featurizer to the given file so it can be reconstructed.

        This function simply calls `SerializableModule.serialize` and saves
        its output to a dictionary. Will also save model parameters if told to.

        Keyword arguments are forwarded to `serialize`.

        Args:
            file (PathLike): File to save model info in.

        """
        properties = self.serialize(**kwargs)
        torch.save(properties, file)

    @classmethod
    def deserialize(cls: Type[SerializableModuleT],
                    properties: Mapping[str, Any],
                    strict: bool = False) -> SerializableModuleT:
        """Instantiate the module from its properties.

        Args:
            properties (Mapping[str, Any]): Module properties. If state_dict
                is in the properties, this method will call
                `torch.nn.Module.load_state_dict` on it.
            strict (bool, optional): See `Serializable.deserialize`.

        Returns:
            SerializableModuleT: Instantiated module.

        """
        properties = {**properties}  # We mutate the dict, so copy it!
        state_dict = properties.pop('state_dict', default=None)
        module = super(cls, cls).deserialize(properties, strict=strict)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module

    @classmethod
    def load(cls: Type[SerializableModuleT], file: PathLike,
             **kwargs: Any) -> SerializableModuleT:
        """Load the module from the given path.

        Keyword arguments are forwarded to `torch.load`.

        Args:
            file (PathLike): File to load module from.

        Returns:
            SerializableModuleT: Loaded module.

        """
        payload = torch.load(file, **kwargs)
        return cls.deserialize(payload)
