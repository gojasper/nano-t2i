import time
from typing import Any, Dict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base import BaseConditioner
from .qwen_embedder_config import QwenEmbedderConfig


class QwenEmbedder(BaseConditioner):
    """This is the QwenEmbedder class which defines the QwenEmbedder model

    Args:

        config (QwenEmbedderConfig): The config class which defines all the required parameters.
    """

    def __init__(self, config: QwenEmbedderConfig):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.version,
            subfolder=config.tokenizer_subfolder,
            revision=config.tokenizer_revision,
        )
        self.tokenizer.padding_side = config.tokenizer_padding_side
        self.transformer = AutoModelForCausalLM.from_pretrained(
            config.version,
            subfolder=config.text_embedder_subfolder,
            revision=config.text_embedder_revision,
        )

        self.tokenizer_truncation = config.tokenizer_truncation
        self.tokenizer_max_length = config.tokenizer_max_length
        self.unconditional_conditioning_value = config.unconditional_conditioning_value
        self.layer_idx = config.layer_idx
        self.returns_attention_mask = config.returns_attention_mask
        self.tokenizer_padding = config.tokenizer_padding

    def on_fit_start(self, device: torch.device | None = None, *args, **kwargs):
        """Called when the training starts"""
        super().on_fit_start(device=device, *args, **kwargs)
        self.transformer = self.transformer.to(device)

    def freeze(self):
        super().freeze()
        self.transformer = self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad = False

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.transformer = self.transformer.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        batch: Dict[str, Any],
        force_zero_embedding: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        """Forward pass of the GemmaEmbedder

        Args:

            batch (Dict[str, Any]): The batch of data
            force_zero_embedding (bool): Whether to force zero embedding.
                This will return an embedding with all entries set to 0. Defaults to False.
        """
        if not isinstance(batch[self.input_key], list):
            batch[self.input_key] = [batch[self.input_key]]
        if (
            force_zero_embedding is not None
            and self.unconditional_conditioning_value is not None
        ):

            unconditional_conditioning = [self.unconditional_conditioning_value] * len(
                batch[self.input_key]
            )
            text = np.where(
                force_zero_embedding,
                unconditional_conditioning,
                batch[self.input_key],
            ).tolist()

        else:
            text = batch[self.input_key]

        for i, text_item in enumerate(text):
            messages = [
                {"role": "user", "content": text_item},
            ]
            text_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            text[i] = text_item

        start_time = time.time()
        batch_encoding = self.tokenizer(
            text,
            truncation=self.tokenizer_truncation,
            max_length=self.tokenizer_max_length,
            padding=self.tokenizer_padding,
            return_tensors="pt",
        )
        end_time = time.time()

        start_time = time.time()
        tokens = batch_encoding["input_ids"].to(device=self.device)
        attention_mask = batch_encoding["attention_mask"].to(device=self.device)

        start_time = time.time()
        outputs = self.transformer(
            tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        end_time = time.time()
        z = outputs.hidden_states[self.layer_idx]

        if (
            force_zero_embedding is not None
            and self.unconditional_conditioning_value is None
        ):
            z = 0 * z
            attention_mask = 0 * attention_mask

        if self.returns_attention_mask:
            output = {self.dim2outputkey[z.dim()]: z, "attention_mask": attention_mask}
        else:
            output = {self.dim2outputkey[z.dim()]: z}

        return output
