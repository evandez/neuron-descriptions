"""Utilities for serializing arbitrary objects."""
from typing import Any, Mapping, Optional, Type, TypeVar

from lv.utils.typing import PathLike

import spacy
import torch
from torch import nn

# Useful type aliases to help with subclassing.
Properties = Mapping[str, Any]
Serialized = Mapping[str, Any]
SerializableTypes = Mapping[str, Any]
ResolvedTypes = Mapping[str, Type['Serializable']]

SerializableT = TypeVar('SerializableT', bound='Serializable')


class Serializable:
    """Mixin that makes an object (loosely) serializable.

    Here, serialization means transforming an object to an equivalent
    python dictionary. The original object should be reconstructable
    from that dictionary alone; however, the serialization procedure need
    not be the same as doing `vars(obj)` even though that is the default
    behavior for this mixin.

    Objects are deserialized by passing the entire dictionary as keyword
    arguments to the constructor. Subclasses should override the `properties`
    method so that the properties exactly reflect the kwargs required by the
    constructor.

    Another feature: if any of the properties are themselves serializable, this
    mixin can handle them! Simply override the `children` function, which
    maps each recursively serializable property name to a unique key
    identifying its python type. These unique keys should be mapped back to
    python types in your override of the @classmethod `resolve`. Additionally,
    this mixin will also automatically serializes some common objects, most
    notably spacy `Language` objects.

    If you don't like this, tough luck. Go use pickle, why don't ya? The use
    case for this mixin is to get the general serialization behavior of pickle
    without tying the payloads to this codebase: you can open the payloads
    anywhere, as they're just dictionaries, and inspect the data yourself or
    even deserialize the object yourself. This is especially useful for
    distributing research models, which will be used across many different
    codebases in settings where the source is heavily, frequently, and
    indiscriminately modified.
    """

    def __init__(self, **_: Any):
        """Initialize the object (necessary for type checking)."""
        super().__init__()

    def properties(self, **_: Any) -> Properties:
        """Return all properties needed to reconstruct the object.

        Properties are basically constructor (kw)args. Keyword arguments are
        arbitrary options for configuring how object is serialized.

        By default, returns dictionary of object fields. Subclasses should
        override this method if different behavior is desired.
        """
        return vars(self)

    def serializable(self, **_: Any) -> SerializableTypes:
        """Return unique keys for each recursively serializable property."""
        return {}

    def serialize(
        self,
        properties_kwargs: Optional[Mapping[str, Any]] = None,
        serializable_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Serialized:
        """Return object serialized as a dictionary.

        Other keyword arguments are unused, but accepted for type-checking
        compatibility.

        Args:
            properties_kwargs (Optional[Mapping[str, Any]], optional): Kwargs
                for call to `properties`. Defaults to None.
            serializable_kwargs (Optional[Mapping[str, Any]], optional): Kwargs
                for call to `serializable`. Defaults to None.

        Raises:
            ValueError: If `serializable` returns keys that refer to
                non-Serializable types.

        Returns:
            Serialized: The serialized object.

        """
        if properties_kwargs is None:
            properties_kwargs = {}
        if serializable_kwargs is None:
            serializable_kwargs = {}

        properties = dict(self.properties(**properties_kwargs))

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
        children = self.serializable(**serializable_kwargs)
        for key, value in properties.items():
            if key in children:
                if not isinstance(value, Serializable):
                    raise ValueError(f'child "{key}" is not serializable '
                                     f'type: {type(value).__name__}')
                properties[key] = value.serialize(**kwargs)

        return {'properties': properties, 'children': children}

    @classmethod
    def deserialize(
        cls: Type[SerializableT],
        serialized: Mapping[str, Any],
        strict: bool = False,
        resolve_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> SerializableT:
        """Deserialize the object from its properties.

        Keyword arguments are passed to recursive `deserialize` calls.

        Args:
            serialized (Mapping[str, Any]): The serialized object.
            strict (bool, optional): If set, die when a recursive deserialize
                is specified but not performed. Defaults to False.
            resolve_kwargs (Optional[Mapping[str, Any]], optional): Kwargs for
                call to `resolve`. Defaults to None.

        Raises:
            KeyError: If `resolve` returns properties not in the payload.

        Returns:
            SerializableT: The deserialized object.

        """
        if resolve_kwargs is None:
            resolve_kwargs = {}

        properties = dict(serialized['properties'])
        children = dict(serialized['children'])

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
        resolved = cls.resolve(children, **resolve_kwargs)
        for key, SerializableType in resolved.items():
            if strict and key not in properties:
                raise KeyError(f'cannot recurse on {key}; not in properties')
            properties[key] = SerializableType.deserialize(
                properties[key], **kwargs)

        deserialized = cls(**properties)
        return deserialized

    @classmethod
    def resolve(cls, children: SerializableTypes, **_: Any) -> ResolvedTypes:
        """Resolve Serializable types for all children."""
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

    def serialize(self,
                  properties_kwargs: Optional[Mapping[str, Any]] = None,
                  serializable_kwargs: Optional[Mapping[str, Any]] = None,
                  state_dict: bool = True,
                  **kwargs: Any) -> Serialized:
        """Serialize the module, including its state dict.

        Keyword arguments are forwarded to `Serializable.serialize`.

        Args:
            properties_kwargs (Optional[Mapping[str, Any]], optional): Kwargs
                for call to `properties`. Defaults to None.
            serializable_kwargs (Optional[Mapping[str, Any]], optional): Kwargs
                for call to `serializable`. Defaults to None.
            state_dict (bool, optional): Include the module's parameters in
                the payload. Defaults to True.

        Returns:
            Serialized: The serialized module.

        """
        serialized = dict(super().serialize(
            properties_kwargs=properties_kwargs,
            serializable_kwargs=serializable_kwargs,
            state_dict=False,
            **kwargs,
        ))
        if state_dict:
            serialized['state_dict'] = self.state_dict()
        return serialized

    def save(self, file: PathLike, **kwargs: Any) -> None:
        """Save the featurizer to the given file so it can be reconstructed.

        This function simply calls `SerializableModule.serialize` and saves
        its output to a dictionary. Will also save model parameters if told to.

        Keyword arguments are forwarded to `serialize`.

        Args:
            file (PathLike): File to save model info in.

        """
        serialized = self.serialize(**kwargs)
        torch.save(serialized, file)

    @classmethod
    def deserialize(cls: Type[SerializableModuleT],
                    serialized: Mapping[str, Any],
                    strict: bool = False,
                    resolve_kwargs: Optional[Mapping[str, Any]] = None,
                    load_state_dict: bool = True,
                    **kwargs: Any) -> SerializableModuleT:
        """Instantiate the module from its properties.

        Keyword arguments are forwarded to `Serializable.deserialize`.

        Args:
            serialized (Mapping[str, Any]): Module properties. If state_dict
                is in the properties, this method will call
                `torch.nn.Module.load_state_dict` on it.
            strict (bool, optional): See `Serializable.deserialize`.
            resolve_kwargs (Optional[Mapping[str, Any]], optional): Kwargs for
                call to `resolve`. Defaults to None.
            load_state_dict (bool, optional): If set, load the serialized
                module parameters. Defaults to True.

        Returns:
            SerializableModuleT: Instantiated module.

        """
        serialized = {**serialized}  # We mutate the dict, so copy it!
        state_dict = serialized.pop('state_dict', None)
        module = super(SerializableModule,
                       cls).deserialize(serialized,
                                        strict=strict,
                                        resolve_kwargs=resolve_kwargs,
                                        load_state_dict=False,
                                        **kwargs)
        if state_dict is not None and load_state_dict:
            module.load_state_dict(state_dict, strict=strict)
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
