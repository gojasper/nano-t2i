import logging
from typing import Any, Dict

import torch

from ..base import BaseConditioner
from .identity_embedder_config import IdentityEmbedderConfig


class IdentityEmbedder(BaseConditioner):
    """This is the IdentityEmbedder class which defines the IdentityEmbedder model

    Args:

        config (ClipEmbedderConfig): The config class which defines all the required parameters.
    """

    def __init__(self, config: IdentityEmbedderConfig):
        BaseConditioner.__init__(self, config)
        self.unconditional_conditioning_value = config.unconditional_conditioning_value

    def on_fit_start(self, device: torch.device | None = None, *args, **kwargs):
        """Called when the training starts"""
        super().on_fit_start(device=device, *args, **kwargs)
        if isinstance(self.unconditional_conditioning_value, torch.Tensor):
            self.unconditional_conditioning_value = (
                self.unconditional_conditioning_value.to(device)
            )
            logging.info(f"Unconditional conditioning value moved to device: {device}")

    def forward(
        self,
        batch: Dict[str, Any],
        force_zero_embedding: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        """
        Forward pass of the ClipEmbedder

        Args:

            batch (Dict[str, Any]): The batch of data
            force_zero_embedding (bool): Whether to force zero embedding.
                This will return an embedding with all entries set to 0. Defaults to False.

        Returns:

            Dict[str, Any]: The output of the embedder. This embedder outputs a 2-dimensional conditioning (type "crossattn")
                and a 1-dimensional conditioning (type "vector") if always_return_pooled is True.
        """

        if force_zero_embedding is not None:
            force_zero_embedding = force_zero_embedding.to(batch[self.input_key])
            if self.unconditional_conditioning_value is not None:

                unconditional_conditioning = torch.cat(
                    [self.unconditional_conditioning_value]
                    * len(batch[self.input_key]),
                    dim=0,
                )

                # pad all dimensions to the same length as batch[self.input_key]
                for i in range(batch[self.input_key].ndim - 1):
                    unconditional_conditioning = torch.nn.functional.pad(
                        unconditional_conditioning,
                        (
                            0,
                            0,
                            0,
                            batch[self.input_key].shape[i + 1]
                            - unconditional_conditioning.shape[i + 1],
                        ),
                    )
                    # unsqueeze unconditional_conditioning to the same length as batch[self.input_key]
                    force_zero_embedding = force_zero_embedding.unsqueeze(-1)

                outputs = unconditional_conditioning * force_zero_embedding + batch[
                    self.input_key
                ] * (1 - force_zero_embedding)
            else:
                outputs = 0 * batch[self.input_key].squeeze()
        else:
            outputs = batch[self.input_key].squeeze()

        if outputs.ndim == 3:
            outputs = outputs[:, : self.config.max_cross_attention_length, :]
        output = {self.dim2outputkey[outputs.dim()]: outputs}
        return output
