from dataclasses import field
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class BaseMapperConfig(BaseConfig):
    """
    Base configuration for mappers.

    Args:

        verbose (bool):
            If True, print debug information. Defaults to False

        key (Optional[str]):
            Key to apply the mapper to. Defaults to None

        output_key (Optional[str]):
            Key to store the output of the mapper. Defaults to None
    """

    verbose: bool = False
    key: Optional[str] = None
    output_key: Optional[str] = None


@dataclass
class KeyRenameMapperConfig(BaseMapperConfig):
    """
    Rename keys in a sample according to a key map

    Args:

        key_map (Dict[str, str]): Dictionary with the old keys as keys and the new keys as values
        condition_key (Optional[str]): Key to use for the condition. Defaults to None
        condition_fn (Optional[Callable[[Any], bool]]): Function to use for the condition to be met so
            the key map is applied. Defaults to None.
        else_key_map (Optional[Dict[str, str]]): Dictionary with the old keys as keys and the new keys as values
            if the condition is not met. Defaults to None *i.e.* the original key will be used.
    """

    key_map: Dict[str, str] = None
    condition_key: Optional[str] = None
    condition_fn: Optional[Callable[[Any], bool]] = None
    else_key_map: Optional[Dict[str, str]] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.key_map is not None, "key_map should be provided"
        assert all(
            isinstance(old_key, str) and isinstance(new_key, str)
            for old_key, new_key in self.key_map.items()
        ), "key_map should be a dictionary with string keys and values"
        if self.condition_key is not None:
            assert self.condition_fn is not None, "condition_fn should be provided"
            assert callable(self.condition_fn), "condition_fn should be callable"
        if self.condition_fn is not None:
            assert self.condition_key is not None, "condition_key should be provided"
            assert isinstance(
                self.condition_key, str
            ), "condition_key should be a string"
        if self.else_key_map is not None:
            assert all(
                isinstance(old_key, str) and isinstance(new_key, str)
                for old_key, new_key in self.else_key_map.items()
            ), "else_key_map should be a dictionary with string keys and values"


@dataclass
class TorchvisionMapperConfig(BaseMapperConfig):
    """
    Apply torchvision transforms to a sample

    Args:

        key (str): Key to apply the transforms to
        transforms (torchvision.transforms): List of torchvision transforms to apply
        transforms_kwargs (Dict[str, Any]): List of kwargs for the transforms
    """

    key: str = "image"
    transforms: List[str] = None
    transforms_kwargs: List[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.transforms is None:
            self.transforms = []
        if self.transforms_kwargs is None:
            self.transforms_kwargs = []
        assert len(self.transforms) == len(
            self.transforms_kwargs
        ), "Number of transforms and kwargs should be same"


@dataclass
class RescaleMapperConfig(BaseMapperConfig):
    """
    Rescale a sample from [0, 1] to [-1, 1]

    Args:

        key (str): Key to rescale
    """

    key: str = "image"


@dataclass
class KeysFromJSONMapperConfig(BaseMapperConfig):
    """
    Get keys from a JSON string and add them to the batch

    Args:

        key (str): Key to extract from the batch
        keys_to_extract (Union[str, List[str]]): Keys to extract from the JSON string
        remove_original (bool): Whether to remove the original key from the batch
        strict (bool): Whether to raise an error if a key is not found in the JSON string
    """

    key: str = "json"
    keys_to_extract: Union[str, List[str]] = None
    remove_original: bool = True
    strict: bool = True


@dataclass
class SelectKeysMapperConfig(BaseMapperConfig):
    """
    Selects keys from the batch

    Args:

        keys (Union[str, List[str]]): Keys to select
    """

    keys: Union[str, List[str]] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.keys is not None, "keys should be provided"


@dataclass
class RemoveKeysMapperConfig(BaseMapperConfig):
    """
    Removes keys from the batch

    Args:

        keys (Union[str, List[str]]): Keys to remove
    """

    keys: Union[str, List[str]] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.keys is not None, "keys should be provided"


@dataclass
class SetValueConfig(BaseMapperConfig):
    """
    Set a value in the batch

    Args:

        key (str): Key to apply the mapper to
        value (Any): Value to set
        value_function (Callable): Function to call to get the value. This is useful to
            set a value dynamically. Example using a function that returns a random value.
        value_function_kwargs (Dict[str, Any]): Keyword arguments for the value function
    """

    value: Any = None
    value_function: Callable = None
    value_function_kwargs: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class DatasetCaptionsConfig:
    caption_keys: List[str]
    caption_probabilities: List[float] = field(default_factory=lambda: [1])

    def __post_init__(self):
        assert len(self.caption_keys) == len(
            self.caption_probabilities
        ), "caption_keys and caption_probabilities must have the same length"
        assert (
            sum(self.caption_probabilities) == 1
        ), "caption_probabilities must sum to 1"


@dataclass
class UrlToDatasetNameConfig(BaseMapperConfig):
    """
    Maps an `url` to a `dataset name` by extracting the dataset name from the url.
    It verifies that the url is valid, by checking that it matches the regex pattern.

    Args:
        url_key (str): Key to use for the url. Defaults to None
        dataset_name_key (str): Key to use for the dataset name. Defaults to None
        regex_pattern (str): Regex pattern to use for the dataset name. Defaults to r"^pipe:aws s3 cp s3://jasper-ai-research/datasets/([^/]+)(?:/([^/]+))?"

    Example:
        url_key: "__url__"
        dataset_name_key: "dataset_name"
        regex_pattern: r"^pipe:aws s3 cp s3://jasper-ai-research/datasets/([^/]+)(?:/([^/]+))?"

        Input:
            batch["__url__"] = "pipe:aws s3 cp s3://jasper-ai-research/datasets/rosemary/000000.tar"
        Output:
            batch["dataset_name"] = "rosemary"
    """

    url_key: str = None
    dataset_name_key: str = None
    regex_pattern: str = (
        r"^pipe:aws s3 cp s3://jasper-ai-research/datasets/([^/]+)(?:/([^/]+))?"
    )


@dataclass
class GetCaptionFromJsonBasedOnNameConfig(BaseMapperConfig):
    """
    This mapper allows to simplify the assignment of the correct captions for
    our internally processed datasets, that were processed differently.

    This mapper is used to assign an existing `deterministic` or `random` caption
    from the caption keys in the mapper configuration.
    By default only a single caption is assigned with probability 1.

    Args:

        dataset_name_key (str): Key to use for the url. Defaults to None
        json_key (str): Key to use for the json. Defaults to None
        output_key (str): Key to use for the output. Defaults to None
        configs (Dict[str, DatasetCaptionsConfig]): Dictionary with the dataset names as keys and the captions configs as values.


    Example:
        dataset_name_key: "dataset_name"
        json_key: "json"
        output_key: "caption"
        configs: {
                "rosemary": DatasetCaptionsConfig(
                    caption_keys=["caption_blip2", "caption_cogvlm"],
                    caption_probabilities=[0.75, 0.25],
                ),
            }

        batch["dataset_name"] = "rosemary"
        batch["json"] = {
            "caption_blip2": "caption_blip2",
            "caption_cogvlm": "caption_cogvlm"
        }

        output:
            batch["caption"] = "caption_blip2" with probability 0.75
            batch["caption"] = "caption_cogvlm" with probability 0.25
    """

    dataset_name_key: str = None
    json_key: str = None
    output_key: str = None
    caption_output_key: str = "caption_name"
    configs: Dict[str, DatasetCaptionsConfig] = field(default_factory=lambda: {})

    def __post_init__(self):
        super().__post_init__()
        assert self.dataset_name_key is not None, "url_key must be provided"
        assert self.json_key is not None, "json_key must be provided"
        assert self.output_key is not None, "output_key must be provided"

        self.datasets_names = list(self.configs.keys())


@dataclass
class SqueezeMapperConfig(BaseMapperConfig):
    """
    Squeeze the input tensor.
    """

    key: str = "image"
    output_key: str = "image"
    dim: int = 0


@dataclass
class BytesToTensorMapperConfig(BaseMapperConfig):
    """
    Convert a Bytes object to a tensor.
    """

    key: str = "image"
    output_key: str = "image"
