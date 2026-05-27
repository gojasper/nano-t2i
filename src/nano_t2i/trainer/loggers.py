import logging
import math
from typing import Any, Dict, List, Tuple

import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid

from ..trainer import TrainingPipeline

logging.basicConfig(level=logging.INFO)


def create_grid_pil_texts(
    texts: List[str],
    n_cols: int = 4,
    image_size: Tuple[int] = (512, 512),
    font_size: int = 40,
    margin: int = 5,
    offset: int = 5,
) -> Image.Image:
    """
    Create a grid of white images containing the given texts.

    Args:
        texts (List[str]): List of strings to be drawn on images.
        n_cols (int): Number of columns in the grid.
        image_size (tuple): Size of the generated images (width, height).
        font_size (int): Font size of the text.
        margin (int): Margin around the text.
        offset (int): Offset between lines.

    Returns:
        PIL.Image: List of generated images as a grid
    """

    images = []
    font = ImageFont.load_default(size=font_size)

    for text in texts:
        img = Image.new("RGB", image_size, color="white")
        draw = ImageDraw.Draw(img)
        margin_ = margin
        offset_ = offset
        for line in wrap_text(
            text=text, draw=draw, max_width=image_size[0] - 2 * margin_, font=font
        ):
            draw.text((margin_, offset_), line, font=font, fill="black")
            offset_ += font_size
        images.append(img)

    # create a pil grid
    n_rows = math.ceil(len(images) / n_cols)
    grid = Image.new(
        "RGB", (n_cols * image_size[0], n_rows * image_size[1]), color="white"
    )
    for i, img in enumerate(images):
        grid.paste(img, (i % n_cols * image_size[0], i // n_cols * image_size[1]))

    return grid


def wrap_text(
    text: str, draw: ImageDraw.Draw, max_width: int, font: ImageFont
) -> List[str]:
    """
    Wrap text to fit within a specified width when drawn.
    It will return to the new line when the text is larger than the max_width.

    Args:
        text (str): The text to be wrapped.
        draw (ImageDraw.Draw): The draw object to calculate text size.
        max_width (int): The maximum width for the wrapped text.
        font (ImageFont): The font used for the text.

    Returns:
        List[str]: List of wrapped lines.
    """
    lines = []
    current_line = ""
    for letter in text:
        if draw.textbbox((0, 0), current_line + letter, font=font)[2] <= max_width:
            current_line += letter
        else:
            lines.append(current_line)
            current_line = letter
    lines.append(current_line)
    return lines


class SampleLogger(Callback):
    """
    General logger for logging samples (images & texts) and metrics.

    Args:
        log_batch_freq (int): The frequency of logging samples from training dataset to the logger regarding the trainer global step. Default is 100.
            If log_batch_freq is set to 0, there is no sample logging during training.
        val_log_batch_freq (int): The frequency of logging samples from validation dataset to the logger regarding the trainer global step. Default is None.
            If the input val_log_batch_freq is None, then val_log_batch_freq is set to have the same value as log_batch_freq.
            If val_log_batch_freq is set to 0, there is no sample logging during validation.
    """

    def __init__(self, log_batch_freq: int = 100, val_log_batch_freq: int = None):
        super().__init__()
        self.log_batch_freq = log_batch_freq
        self.val_log_batch_freq = (
            val_log_batch_freq if val_log_batch_freq is not None else log_batch_freq
        )
        ## stored_training_outputs : training outputs averaged through all accumulation steps
        self.stored_training_outputs = {}
        ## stored_validation_outputs : validation outputs averaged through all accumulation steps
        self.stored_validation_outputs = {}

    def update_stored_outputs(
        self,
        stored_outputs: Dict[str, Any],
        outputs: Dict[str, Any],
        pl_module: TrainingPipeline,
        split: str = "train",
    ) -> Dict[str, Any]:
        if pl_module.num_optimizers > 1:
            assert (
                "loss" not in outputs
            ), "The loss should not be logged for multiple optimizers."
        elif split == "train":
            # detach the loss in case of one optimizer
            outputs["loss"] = outputs["loss"].detach().cpu()

        num_accumulation = (
            pl_module.num_train_accumulation
            if split == "train"
            else pl_module.num_val_accumulation
        )
        if stored_outputs:
            # update stored_outputs with current outputs if stored_outputs is not empty
            assert (
                stored_outputs.keys() == outputs.keys()
            ), "Stored outputs and current outputs should have the same keys."
            # - for loss-like quantities : add up the current ouputs with proper accumulation rescaling
            #    -> /!!\ exception for "loss" quantity with one optimizer in 'train' setting : it is already rescaled by Lightning
            # - for other quantities (like batch_idx) : replace the stored outputs
            stored_outputs = {
                k: (
                    stored_outputs[k] + v
                    if k == "loss"
                    and split
                    == "train"  # we are sure to have 1 optimizer with initial assert
                    else stored_outputs[k] + v / num_accumulation if "loss" in k else v
                )
                for k, v in outputs.items()
            }
        else:
            # create the stored outputs with proper accumulation rescaling
            stored_outputs = {
                k: (
                    v
                    if k == "loss"
                    and split
                    == "train"  # we are sure to have 1 optimizer with initial assert
                    else v / num_accumulation if "loss" in k else v
                )
                for k, v in outputs.items()
            }
        return stored_outputs

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        # update stored training outputs with current outputs
        self.stored_training_outputs = self.update_stored_outputs(
            self.stored_training_outputs, outputs, pl_module, split="train"
        )

        # 1. log stored_training_outputs at every trainer global step, once full accumulation is reached
        if (
            pl_module.training_forward_pass_counter % pl_module.num_train_accumulation
            == 0
        ):
            # process the logs
            self._process_logs(trainer, self.stored_training_outputs, split="train")
            # restore stored_training_outputs
            self.stored_training_outputs = {}

        if self.log_batch_freq > 0:
            # 2. log samples always at the end of training accumulation:
            # - at the first step
            # - at the frequency fixed by log_batch_freq
            is_logging_step = (pl_module.global_step == 1) or (
                pl_module.global_step % self.log_batch_freq == 0
            )
            is_full_accumulation = (
                pl_module.training_forward_pass_counter
                % pl_module.num_train_accumulation
                == 0
            )
            if is_logging_step and is_full_accumulation:
                self.log_samples(
                    trainer, pl_module, outputs, batch, batch_idx, split="train"
                )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        # update stored validation outputs with current outputs
        self.stored_validation_outputs = self.update_stored_outputs(
            self.stored_validation_outputs, outputs, pl_module, split="val"
        )

        # 1. log stored_validation_outputs at every trainer global step, once full accumulation is reached
        if (
            pl_module.validation_forward_pass_counter % pl_module.num_val_accumulation
            == 0
        ):
            # process the logs
            self._process_logs(trainer, self.stored_validation_outputs, split="val")
            # restore stored_validation_outputs
            self.stored_validation_outputs = {}

        if self.val_log_batch_freq > 0:
            # 2. log samples always at the end of validation accumulation:
            # - at the first step
            # - at the frequency fixed by val_log_batch_freq
            is_logging_step = (pl_module.global_step == 1) or (
                pl_module.global_step % self.val_log_batch_freq == 0
            )
            is_full_accumulation = (
                pl_module.validation_forward_pass_counter
                % pl_module.num_val_accumulation
                == 0
            )
            if is_logging_step and is_full_accumulation:
                self.log_samples(
                    trainer, pl_module, outputs, batch, batch_idx, split="val"
                )

    @torch.no_grad()
    def log_samples(
        self,
        trainer: Trainer,
        pl_module: TrainingPipeline,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        split: str = "train",
    ) -> None:

        if hasattr(pl_module, "log_samples"):
            is_training = pl_module.training
            if is_training:
                pl_module.eval()

            logs = pl_module.log_samples(batch)
            self._process_logs(trainer, logs, split=split)

            if is_training:
                pl_module.train()
        else:
            logging.warning(
                "log_img method not found in LightningModule. Skipping image logging."
            )

    @rank_zero_only
    def _process_logs(
        self, trainer, logs: Dict[str, Any], rescale=True, split="train"
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "Function to process logs has to be implemented for each type of logger."
        )


class WandbSampleLogger(SampleLogger):
    """
    Logger for logging samples and metrics to wandb.

    Args:
        log_batch_freq (int): The frequency of logging samples from training dataset to wandb regarding the trainer global step. Default is 100.
            If log_batch_freq is set to 0, there is no sample logging during training.
        val_log_batch_freq (int): The frequency of logging samples from validation dataset to wandb regarding the trainer global step. Default is None.
            If the input val_log_batch_freq is None, then val_log_batch_freq is set to have the same value as log_batch_freq.
            If val_log_batch_freq is set to 0, there is no sample logging during validation.
    """

    @rank_zero_only
    def _process_logs(
        self, trainer, logs: Dict[str, Any], rescale=True, split="train"
    ) -> Dict[str, Any]:
        for key, value in logs.items():
            logging.debug(f"Logging {key} samples")

            if isinstance(value, torch.Tensor):
                logging.debug(f"Logging {key} samples as tensor")
                value = value.detach().cpu()
                if value.dim() == 4:
                    images = value
                    if rescale:
                        images = (images + 1.0) / 2.0
                    grid = make_grid(images, nrow=4)
                    grid = grid.permute(1, 2, 0)
                    grid = grid.mul(255).clamp(0, 255).to(torch.uint8)
                    logs[key] = grid.numpy()
                    trainer.logger.experiment.log(
                        {f"{key}/{split}": [wandb.Image(Image.fromarray(logs[key]))]},
                        step=trainer.global_step,
                    )

                # Scalar tensor
                if value.dim() == 1 or value.dim() == 0:
                    value = value.float().numpy()
                    trainer.logger.experiment.log(
                        {f"{key}/{split}": value}, step=trainer.global_step
                    )

            # list of string (e.g. text)
            if isinstance(value, list):
                logging.debug(f"Logging {key} samples as list")
                if isinstance(value[0], str):
                    pil_image_texts = create_grid_pil_texts(value)
                    wandb_image = wandb.Image(pil_image_texts)
                    trainer.logger.experiment.log(
                        {f"{key}/{split}": [wandb_image]},
                        step=trainer.global_step,
                    )

            # dict of tensors (e.g. metrics)
            if isinstance(value, dict):
                logging.debug(f"Logging {key} samples as dict")
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        value[k] = v.detach().cpu().numpy()
                trainer.logger.experiment.log(
                    {f"{key}/{split}": value}, step=trainer.global_step
                )

            if isinstance(value, int) or isinstance(value, float):
                logging.debug(f"Logging {key} samples as int or float")
                trainer.logger.experiment.log(
                    {f"{key}/{split}": value}, step=trainer.global_step
                )

        return logs
