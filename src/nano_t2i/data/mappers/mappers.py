import io
import json
import logging
import random
import re
from typing import Any, Callable, Dict, List, Tuple

import torch
from torchvision import transforms

from .base import BaseMapper
from .mappers_config import (
    BytesToTensorMapperConfig,
    GetCaptionFromJsonBasedOnNameConfig,
    KeyRenameMapperConfig,
    KeysFromJSONMapperConfig,
    RescaleMapperConfig,
    SelectKeysMapperConfig,
    SetValueConfig,
    SqueezeMapperConfig,
    TorchvisionMapperConfig,
    UrlToDatasetNameConfig,
)


class KeyRenameMapper(BaseMapper):
    """
    Rename keys in a sample according to a key map

    Args:

        config (KeyRenameMapperConfig): Configuration for the mapper

    Examples
    ########

    1. Rename keys in a sample according to a key map

    .. code-block:: python

        from nano_t2i.data.mappers import KeyRenameMapper, KeyRenameMapperConfig

        config = KeyRenameMapperConfig(
            key_map={"old_key": "new_key"}
        )

        mapper = KeyRenameMapper(config)

        sample = {"old_key": 1}
        new_sample = mapper(sample)
        print(new_sample)  # {"new_key": 1}

    2. Rename keys in a sample according to a key map and a condition key

    .. code-block:: python

        from nano_t2i.data.mappers import KeyRenameMapper, KeyRenameMapperConfig

        config = KeyRenameMapperConfig(
            key_map={"old_key": "new_key"},
            condition_key="condition",
            condition_fn=lambda x: x == 1
        )

        mapper = KeyRenameMapper(config)

        sample = {"old_key": 1, "condition": 1}
        new_sample = mapper(sample)
        print(new_sample)  # {"new_key": 1}

        sample = {"old_key": 1, "condition": 0}
        new_sample = mapper(sample)
        print(new_sample)  # {"old_key": 1}

    ```
    """

    def __init__(self, config: KeyRenameMapperConfig):
        super().__init__(config)
        self.key_map = config.key_map
        self.condition_key = config.condition_key
        self.condition_fn = config.condition_fn
        self.else_key_map = config.else_key_map

    def __call__(self, batch: Dict[str, Any], *args, **kwrags):
        if self.condition_key is not None:
            condition_key = batch[self.condition_key]
            if self.condition_fn(condition_key):
                for old_key, new_key in self.key_map.items():
                    if old_key in batch:
                        batch[new_key] = batch.pop(old_key)

            elif self.else_key_map is not None:
                for old_key, new_key in self.else_key_map.items():
                    if old_key in batch:
                        batch[new_key] = batch.pop(old_key)

        else:
            for old_key, new_key in self.key_map.items():
                if old_key in batch:
                    batch[new_key] = batch.pop(old_key)
        return batch


class TorchvisionMapper(BaseMapper):
    """
    Apply torchvision transforms to a sample

    Args:

        config (TorchvisionMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: TorchvisionMapperConfig):
        super().__init__(config)
        chained_transforms = []
        for transform, kwargs in zip(config.transforms, config.transforms_kwargs):
            transform = getattr(transforms, transform)
            chained_transforms.append(transform(**kwargs))
        self.transforms = transforms.Compose(chained_transforms)

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        if self.key in batch:
            batch[self.output_key] = self.transforms(batch[self.key])
        else:
            logging.warning(f"Key {self.key} not found in batch in TorchvisionMapper")
        return batch


class RescaleMapper(BaseMapper):
    """
    Rescale a sample from [0, 1] to [-1, 1]

    Args:

        config (RescaleMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: RescaleMapperConfig):
        super().__init__(config)

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        if isinstance(batch[self.key], list):
            tmp = []
            for i, image in enumerate(batch[self.key]):
                tmp.append(2 * image - 1)
            batch[self.output_key] = tmp
        else:
            batch[self.output_key] = 2 * batch[self.key] - 1
        return batch


class KeysFromJSONMapper(BaseMapper):
    """
    Get keys from a JSON string and add them to the batch

    Args:

        config (KeysFromJSONMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: KeysFromJSONMapperConfig):
        super().__init__(config)
        keys_to_extract = config.keys_to_extract
        self.remove_original = config.remove_original
        self.strict = config.strict

        if isinstance(keys_to_extract, str):
            self.keys_to_extract = [keys_to_extract]
        else:
            self.keys_to_extract = keys_to_extract

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        assert self.key in batch, f"Key {self.key} not in batch"
        if isinstance(batch[self.key], str):
            decoded_json = json.loads(batch[self.key])
        elif isinstance(batch[self.key], dict):
            decoded_json = batch[self.key]
        else:
            raise ValueError(f"Key {self.key} is not a string or a dict")

        for key in self.keys_to_extract:
            try:
                batch[key] = decoded_json[key]
            except KeyError as e:
                # If the key is not found, raise an error or continue
                if self.strict:
                    logging.error(f"Key {key} not found in JSON")
                    raise e
                # If the key is not found, continue
                else:
                    logging.debug(f"Key {key} not found in JSON")
                    continue
        if self.remove_original:
            logging.debug(f"Removing original key {self.key}")
            del batch[self.key]
        return batch


class SelectKeysMapper(BaseMapper):
    """
    Select keys from a sample and remove the rest

    Args:

        config (SelectKeysMapperConfig): Configuration for the mapper
    """

    def __init__(self, config: SelectKeysMapperConfig):
        super().__init__(config)
        keys = config.keys
        if isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = keys

    def __call__(self, batch: Dict[str, Any], *args, **kwrags) -> Dict[str, Any]:
        return {key: batch[key] for key in self.keys}


class SetValueMapper(BaseMapper):
    """SetValueMapper

    Set a value to a key in the batch

    Args:
        config (SetValueConfig): Configuration for the mapper

    ## Examples:

    ### Example 1: With a static value
        .. code-block:: python
            mapper = SetValueMapper(
                config=SetValueConfig(
                    key="key",
                    value=1,
                )
            )

            batch = {"key": "value"}
            result = mapper(batch)
            print(result)
            # the output will be {"key": 1}


    ### Example 2: With a dynamic value
        .. code-block:: python
            mapper = SetValueMapper(
                config=SetValueConfig(
                    key="key",
                    value_function=np.random.choice,
                    value_function_kwargs={"a": [1, 2], "p": [0.5, 0.5]},
                )
            )

            batch = {"key": "value"}
            result = mapper(batch)
            print(result)
            # the output will be {"key": 1} with 50% probability and {"key": 2} with 50% probability
    """

    def __init__(self, config: SetValueConfig):
        super().__init__(config)
        self.value = config.value
        if config.value_function is not None:
            self.value = config.value_function
        self.key = config.key
        self.value_function_kwargs = config.value_function_kwargs

    def __call__(self, batch: Dict[str, Any], *args, **kwrags):
        if self.value is not None:
            if isinstance(self.value, Callable):
                batch[self.key] = self.value(**self.value_function_kwargs)
            else:
                batch[self.key] = self.value
        return batch


class UrlToDatasetNameMapper(BaseMapper):
    """UrlToDatasetNameMapper

    Maps an `url` to a `dataset name` by extracting the dataset name from the url.
    It verifies that the url is valid, by checking that it matches the regex pattern.

    Args:
        url_key (str): Key to use for the url. Defaults to None
        dataset_name_key (str): Key to use for the dataset name. Defaults to None
        regex_pattern (str): Regex pattern to use for the dataset name. Defaults to r"^pipe:aws s3 cp s3://jasper-ai-research/datasets/([^/]+)/([^/]+)"

    Example:
        url_key: "__url__"
        dataset_name_key: "dataset_name"
        regex_pattern: r"^pipe:aws s3 cp s3://jasper-ai-research/datasets/([^/]+)/([^/]+)"

        Input:
            batch["__url__"] = "pipe:aws s3 cp s3://jasper-ai-research/datasets/rosemary/000000.tar"
        Output:
            batch["dataset_name"] = "rosemary"
    """

    def __init__(self, config: UrlToDatasetNameConfig):
        super().__init__(config)
        self.url_key = config.url_key
        self.dataset_name_key = config.dataset_name_key
        self.regex_pattern = config.regex_pattern

    def __call__(self, batch: Dict[str, Any], *args, **kwrags):
        assert self.url_key in batch, f"Key {self.url_key} not in batch"
        url = batch[self.url_key]
        assert isinstance(url, str), f"Url {url} is not a string"

        # Requires to match the regex pattern
        match = re.match(self.regex_pattern, url)
        if match:
            batch[self.dataset_name_key] = match.group(1)
        else:
            raise ValueError(
                f"Url {url} does not match the required pattern {self.regex_pattern}"
            )
        return batch


class GetCaptionFromJsonBasedOnNameMapper(BaseMapper):
    """
    GetCaptionFromJsonBasedOnNameMapper

    This mapper allows to simplify the assignment of the correct captions for
    our internally processed datasets, that were processed differently.

    This mapper is used to assign an existing `deterministic` or `random` caption
    from the caption keys in the mapper configuration.
    By default only a single caption is assigned with probability 1.

    Args:
        config (GetCaptionFromJsonBasedOnNameConfig): Configuration for the mapper

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

    def __init__(self, config: GetCaptionFromJsonBasedOnNameConfig):
        super().__init__(config)
        self.dataset_name_key = config.dataset_name_key
        self.json_key = config.json_key
        self.output_key = config.output_key
        self.caption_output_key = config.caption_output_key
        self.dataset_captions_configs = config.configs

    def _randomizer(
        self, batch, captions_keys: List[str], probabilities: List[float]
    ) -> str:
        available_keys = []
        available_probabilities = []
        for key, p in zip(captions_keys, probabilities):
            if key in batch[self.json_key]:
                available_keys.append(key)
                available_probabilities.append(p)
            elif p == 0.0:
                continue
            elif self.verbose:
                logging.warning(
                    f"Key {key} not in batch for url {batch['__url__']}. Probability for this key will be set to 0."
                )

        if len(available_keys) == 0:
            raise ValueError(
                f"No available keys found for url {batch[self.dataset_name_key]}. Available keys: {batch[self.json_key].keys()}"
            )

        available_probabilities = [
            p / sum(available_probabilities) for p in available_probabilities
        ]
        # assert (
        #     key in batch[self.json_key]
        # ), f"Key {key} not in batch for url {batch[self.dataset_name_key]}. Available keys: {batch[self.json_key].keys()}"

        # convert probabilities to weights
        selected_index = random.choices(
            range(len(available_keys)), weights=available_probabilities
        )[0]
        selected_caption_key = available_keys[selected_index]
        selected_caption = batch[self.json_key][selected_caption_key]
        logging.debug(
            f"GetCaptionFromJsonMapper: Selected caption: {selected_caption_key}"
        )
        return selected_caption, selected_caption_key

    def get_caption(self, dataset_name: str, batch: Dict[str, Any]) -> Tuple[str, str]:
        caption = None

        assert (
            dataset_name in self.dataset_captions_configs
        ), f"Dataset name {dataset_name} not in dataset_captions_configs"
        dataset_captions_config = self.dataset_captions_configs[dataset_name]
        caption, caption_key = self._randomizer(
            batch=batch,
            captions_keys=dataset_captions_config.caption_keys,
            probabilities=dataset_captions_config.caption_probabilities,
        )
        batch[self.output_key] = caption
        batch[self.caption_output_key] = caption_key
        return caption

    def __call__(self, batch: Dict[str, Any], *args, **kwrags):
        assert (
            self.dataset_name_key in batch
        ), f"Key {self.dataset_name_key} not in batch. Available keys: {batch.keys()}. You might want to use the UrlToDatasetNameMapper first to get the dataset name from the webdataset urls"
        assert self.json_key in batch, f"Key {self.json_key} not in batch"

        dataset_name = batch[self.dataset_name_key]
        caption = self.get_caption(dataset_name=dataset_name, batch=batch)

        batch[self.output_key] = caption
        return batch


class SqueezeMapper(BaseMapper):
    """
    Squeeze the input tensor.
    """

    def __init__(self, config: SqueezeMapperConfig):
        super().__init__(config)
        self.image_key = config.key
        self.output_key = config.output_key
        self.dim = config.dim

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch[self.image_key] = batch[self.image_key].squeeze(self.dim)
        return batch


class BytesToTensorMapper(BaseMapper):
    """
    Convert a Bytes object to a tensor.
    """

    def __init__(self, config: BytesToTensorMapperConfig):
        super().__init__(config)
        self.image_key = config.key
        self.output_key = config.output_key

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch[self.output_key] = torch.load(
            io.BytesIO(batch[self.key]), map_location="cpu", weights_only=False
        )
        return batch
