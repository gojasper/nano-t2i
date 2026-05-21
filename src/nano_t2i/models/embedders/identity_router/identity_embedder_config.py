from typing import Any, Optional

from pydantic.dataclasses import dataclass

from ..base import BaseConditionerConfig


@dataclass
class IdentityEmbedderConfig(BaseConditionerConfig):
    """This is the ClipEmbedderConfig class which defines all the useful parameters to instantiate the model

    Args:

        max_cross_attention_length (int): The maximum length of the cross attention. Defaults to 256.
        unconditional_conditioning_value (Optional[Any]): The value pass to the embedder for unconditional conditioning.
            If None, the embedding will be set to 0. Defaults to None.
    """

    max_cross_attention_length: int = 256
    unconditional_conditioning_value: Optional[Any] = None
