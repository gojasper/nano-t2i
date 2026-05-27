from typing import Optional

from pydantic.dataclasses import dataclass

from ..base import BaseConditionerConfig


@dataclass
class QwenEmbedderConfig(BaseConditionerConfig):
    """This is the QwenEmbedderConfig class which defines all the useful parameters to instantiate the model

    Args:

        version (str): The version of the model on HF Hub. Defaults to "Qwen/Qwen3-4B-Instruct-2507".
        text_embedder_subfolder (str): The subfolder for the text embedder if loaded from an other model. Defaults to "".
        tokenizer_subfolder (str): The subfolder for the tokenizer if loaded from an other model. Defaults to "".
        text_embedder_revision (str): The revision of the text embedder. Defaults to "main".
        tokenizer_revision (str): The revision of the tokenizer. Defaults to "main".
        input_key (str): The key for the input. Defaults to "text".
        tokenizer_truncation (bool): Whether to truncate the tokenizer. Defaults to True.
        tokenizer_max_seq_length (int): The maximum sequence length of the tokenizer. Defaults to 256.
        layer_idx (int): The layer index to use for the embedding. Defaults to -1.
        tokenizer_padding_side (str): The padding side of the tokenizer. Defaults to "right".
        unconditional_conditioning_value (Optional[str]): The value pass to the embedder for unconditional conditioning.
            If None, the embedding will be set to 0. Defaults to None.
        returns_attention_mask (bool): Whether to return the attention mask. Defaults to False.
    """

    version: str = "Qwen/Qwen3-4B-Instruct-2507"
    text_embedder_subfolder: str = ""
    tokenizer_subfolder: str = ""
    text_embedder_revision: str = "main"
    tokenizer_revision: str = "main"
    input_key: str = "text"
    tokenizer_truncation: bool = True
    tokenizer_max_length: Optional[int] = 256
    layer_idx: int = -2
    tokenizer_padding_side: str = "right"
    tokenizer_padding: str = "max_length"
    unconditional_conditioning_value: Optional[str] = None
    returns_attention_mask: bool = False
