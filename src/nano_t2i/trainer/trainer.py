import importlib
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist

from ..data.mappers import MapperWrapper
from ..models.base.base_model import BaseModel
from .training_config import TrainingConfig

logging.basicConfig(level=logging.INFO)
import os


class TrainingPipeline(pl.LightningModule):
    """
    Main Training Pipeline class for ClipDrop.

    Args:

        model (BaseModel): The model to train
        pipeline_config (TrainingConfig): The configuration for the training pipeline
        verbose (bool): Whether to print logs in the console. Default is False.
    """

    def __init__(
        self,
        model: BaseModel,
        pipeline_config: TrainingConfig,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.pipeline_config = pipeline_config
        self.log_samples_model_kwargs = pipeline_config.log_samples_model_kwargs
        self.num_optimizers = len(pipeline_config.optimizers_name)
        self.opts_cum_num_backward_steps = np.cumsum(
            np.array(pipeline_config.optimizers_num_backward_steps)
        )
        self.optimizer1_start_step = pipeline_config.optimizer1_start_step
        self.optimizer1_warmup_steps = pipeline_config.optimizer1_warmup_steps
        self.optimizer1_rewarm_steps = pipeline_config.optimizer1_rewarm_steps
        self.optimizer1_rewarm_num_steps = pipeline_config.optimizer1_rewarm_num_steps
        # self.opts_cum_num_backward_steps counts the accumulated global trainer steps when switching optimizers

        if self.optimizer1_rewarm_steps is not None:
            self.warmup_step_counter = 0

        # save hyperparameters.
        self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters({"model_config": model.config.to_dict()})

        # logger.
        self.verbose = verbose

        # setup logging.
        log_keys = pipeline_config.log_keys
        if isinstance(log_keys, str):
            log_keys = [log_keys]
        if log_keys is None:
            log_keys = []
        self.log_keys = log_keys

        ############################## gradient & validation accumulation ##########################
        self.num_train_accumulation = pipeline_config.accumulate_grad_batches
        self.num_val_accumulation = pipeline_config.limit_val_batches
        ## training_forward_pass_counter : number of cumulated forward passes at training stage
        ## it is incremented by 1 every time a training batch is evaluated, it is restarted every time we fit the model
        self.training_forward_pass_counter = 0
        ## validation_forward_pass_counter : number of cumulated forward passes at validation stage
        ## it is incremented by 1 every time a validation batch is evaluated, it is restarted at the beginning of every validation stage
        self.validation_forward_pass_counter = 0

    def on_fit_start(self) -> None:
        if self.global_rank == 0:
            logging.info("START on_fit_start")
        self.model.on_fit_start(device=self.device)
        self.training_forward_pass_counter = 0
        if self.global_rank == 0:
            self.timer = time.perf_counter()
        if self.global_rank == 0:
            logging.info("END on_fit_start")
        logging.info(
            f"Device rank: {self.global_rank}, node: {os.environ['SLURMD_NODENAME']}, device: {self.device}"
        )

    def on_validation_epoch_start(self) -> None:
        logging.debug("START on_validation_start")
        self.validation_forward_pass_counter = 0
        logging.debug("END on_validation_start")

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def on_after_backward(self, *args, **kwargs) -> None:
        if self.global_rank == 0:
            logging.debug("on_after_backward")
        self.model.on_after_backward(
            forward_count=self.training_forward_pass_counter, *args, **kwargs
        )

    def on_train_batch_end(
        self, outputs: Dict[str, Any], batch: Any, batch_idx: int
    ) -> None:
        if self.global_rank == 0:
            logging.debug("START on_train_batch_end")
        if self.global_rank == 0:
            logging.debug("on_train_batch_end")
        self.model.on_train_batch_end(batch)

        average_time_frequency = 10
        if self.global_rank == 0 and batch_idx % average_time_frequency == 0:
            delta = time.perf_counter() - self.timer
            logging.debug(
                f"Average time per batch {batch_idx} took {delta / (batch_idx + 1)} seconds"
            )
        if self.global_rank == 0:
            logging.debug("END on_train_batch_end")

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """
        Setup optimizers and learning rate schedulers.
        """
        optimizers = []
        for i in range(len(self.pipeline_config.optimizers_name)):
            lr = self.pipeline_config.learning_rates[i]
            param_list = []
            n_params = 0
            param_list_ = {"params": []}
            for name, param in self.model.named_parameters():
                for regex in self.pipeline_config.trainable_params[i]:
                    pattern = re.compile(regex)
                    if re.match(pattern, name):
                        if param.requires_grad:
                            param_list_["params"].append(param)
                            n_params += param.numel()

            param_list.append(param_list_)

            logging.info(
                f"Number of trainable parameters for optimizer {i}: {n_params}"
            )

            optimizer_cls = getattr(
                importlib.import_module("torch.optim"),
                self.pipeline_config.optimizers_name[i],
            )
            optimizer = optimizer_cls(
                param_list, lr=lr, **self.pipeline_config.optimizers_kwargs[i]
            )
            optimizers.append(optimizer)

        if len(optimizers) > 1:
            self.automatic_optimization = False

        self.optims = optimizers
        schedulers_config = self.configure_lr_schedulers()

        for name, param in self.model.named_parameters():
            set_grad_false = True
            for regexes in self.pipeline_config.trainable_params:
                for regex in regexes:
                    pattern = re.compile(regex)
                    if re.match(pattern, name):
                        if param.requires_grad:
                            set_grad_false = False
            if set_grad_false:
                param.requires_grad = False

        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logging.info(f"Number of trainable parameters: {num_trainable_params}")

        schedulers_config = self.configure_lr_schedulers()

        if schedulers_config is None:
            return optimizers

        return optimizers, [
            schedulers_config_ for schedulers_config_ in schedulers_config
        ]

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ):
        assert gradient_clip_algorithm in ("norm", None), gradient_clip_algorithm
        if gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)

    def configure_lr_schedulers(self) -> List[Dict[str, Any]]:
        schedulers_config = []
        for i in range(len(self.pipeline_config.lr_schedulers_name)):
            if self.pipeline_config.lr_schedulers_name[i] is None:
                scheduler = None
                schedulers_config.append(scheduler)
            else:
                scheduler_cls = getattr(
                    importlib.import_module("torch.optim.lr_scheduler"),
                    self.pipeline_config.lr_schedulers_name[i],
                )
                scheduler = scheduler_cls(
                    self.optims[i],
                    **self.pipeline_config.lr_schedulers_kwargs[i],
                )
                lr_scheduler_config = {
                    "scheduler": scheduler,
                    "interval": self.pipeline_config.lr_schedulers_interval[i],
                    "monitor": "val_loss",
                    "frequency": self.pipeline_config.lr_schedulers_frequency[i],
                }
                schedulers_config.append(lr_scheduler_config)

        if all([scheduler is None for scheduler in schedulers_config]):
            return None

        return schedulers_config

    def compute_grad_norm_2_optimizers(self, idx_opt: int = 0):
        model, key = (
            (self.model.student_denoiser, "student_denoiser")
            if idx_opt == 0
            else (self.model.discriminator, "discriminator")
        )
        total_norm = torch.tensor(0.0, device=next(model.parameters()).device)

        for p in model.parameters():
            if p.is_leaf and p.requires_grad and p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm**2

        if key == "student_denoiser":
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)

        total_norm = total_norm.sqrt().float().item()

        if self.global_rank == 0:
            logging.info(f"grad norm for network {key} :{total_norm}")
        return key, total_norm

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> dict:
        logging.debug("START training_step")
        logging.debug("--------------------------------")
        if self.automatic_optimization:
            model_forward_start_time = time.perf_counter()
            model_output = self.model(
                train_batch,
                step=self.global_step,
                forward_count=self.training_forward_pass_counter,
            )
            model_forward_end_time = time.perf_counter()
            logging.debug(
                f"Model forward time: {model_forward_end_time - model_forward_start_time} seconds"
            )
            self.training_forward_pass_counter += 1

            loss = model_output["loss"]
            logging.debug(
                f"loss: {loss}, global_rank:{self.global_rank}, local_rank:{self.local_rank}"
            )
            self.log(
                "loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                logger=False,
            )

            # get all loss-liked quantities
            outputs = {
                k: v.detach().cpu()
                for k, v in model_output.items()
                if k != "loss" and "loss" in k
            }
            outputs.update(
                {
                    "batch_idx": batch_idx,
                    **self.model.additional_training_logs(train_batch, batch_idx),
                    # "ce_loss": model_output["ce_loss"],
                    # "latent_loss": model_output["latent_loss"],
                }
            )

            # add "loss" attribute only when there is one optimizer
            outputs["loss"] = loss

        # manual optim for multiple optimizers : same optimizer for a given global step
        else:

            if self.global_step < self.optimizer1_start_step:
                idx_opt = 0
                logging.info(
                    f"Global step {self.global_step} < optimizer1_start_step {self.optimizer1_start_step}, using optimizer 0"
                )

            elif (
                self.global_step
                < self.optimizer1_warmup_steps + self.optimizer1_start_step
            ):
                logging.info(
                    f"Global step {self.global_step} < optimizer1_warmup_steps + optimizer1_start_step {self.optimizer1_warmup_steps + self.optimizer1_start_step}, warming up optimizer 1"
                )
                idx_opt = 1
            elif self.optimizer1_rewarm_steps > 0 and (
                self.global_step % self.optimizer1_rewarm_steps
                < self.optimizer1_rewarm_num_steps
            ):
                logging.info(
                    f"Global step {self.global_step} % optimizer1_rewarm_steps {self.optimizer1_rewarm_steps} < optimizer1_rewarm_num_steps {self.optimizer1_rewarm_num_steps}, rewarming optimizer 1"
                )
                idx_opt = 1
                self.warmup_step_counter += 1
            else:
                # select the adequate optimizer
                idx_opt = np.searchsorted(
                    self.opts_cum_num_backward_steps,
                    (self.global_step % self.opts_cum_num_backward_steps[-1]) + 1,
                    side="left",
                )
                self.warmup_step_counter = 0

            model_output = self.model(
                train_batch,
                step=self.global_step,
                forward_count=self.training_forward_pass_counter,
                opt_idx=idx_opt,
            )
            self.training_forward_pass_counter += 1

            loss = model_output["loss"]
            logging.debug(
                f"loss: {loss}, global_rank:{self.global_rank}, local_rank:{self.local_rank}"
            )

            # get all loss-liked quantities
            outputs = {
                k: v.detach().cpu()
                for k, v in model_output.items()
                if k != "loss" and "loss" in k
            }
            outputs.update(
                {
                    "batch_idx": batch_idx,
                    **self.model.additional_training_logs(train_batch, batch_idx),
                    # "ce_loss": model_output["ce_loss"],
                    # "latent_loss": model_output["latent_loss"],
                }
            )

            if self.global_rank == 0:
                logging.info(
                    f"opts_cum_num_backward_steps: {self.opts_cum_num_backward_steps}"
                )
                logging.info(f"global_step: {self.global_step}")
                logging.info(f"loss for optimizer {idx_opt}: {loss[idx_opt]}")

            current_opt = self.optimizers()[idx_opt]
            self.toggle_optimizer(current_opt)

            # get the adequate loss with accumulation rescaling and do the backward
            self.manual_backward(loss[idx_opt] / self.num_train_accumulation)

            # untoggle before the optimizer step
            self.untoggle_optimizer(current_opt)

            if self.training_forward_pass_counter % self.num_train_accumulation == 0:
                # do the optimizer step once full accumulation is reached
                current_opt.step()
                current_opt.zero_grad()

        logging.debug("END training_step")
        return outputs

    def validation_step(self, val_batch: Dict[str, Any], val_idx: int) -> dict:
        assert (
            self.num_val_accumulation > 0
        ), "The TrainingPipeline argument 'limit_val_batches' has to be the same as the Lightning Trainer argument."

        logging.debug("START validation_step")
        model_output = self.model(
            val_batch,
            device=self.device,
            step=self.global_step,
            forward_count=self.validation_forward_pass_counter,
        )
        self.validation_forward_pass_counter += 1

        loss = model_output["loss"]
        logging.debug(
            f"loss: {loss}, global_rank:{self.global_rank}, local_rank:{self.local_rank}"
        )

        # get all loss-liked quantities
        outputs = {
            k: v.detach().cpu()
            for k, v in model_output.items()
            if k != "loss" and "loss" in k
        }

        # add the loss only if there is one optimizer <-> in case of automatic optimization
        if self.automatic_optimization:
            outputs["loss"] = loss.detach().cpu()

        logging.debug("END validation_step")
        return outputs

    def log_samples(self, batch: Dict[str, Any]):
        if self.global_rank == 0:
            logging.info("START log_samples")
        logs = self.model.log_samples(
            batch,
            **self.log_samples_model_kwargs,
        )

        if logs is not None:
            N = min([len(logs[keys]) for keys in logs])
        else:
            N = 0

        # Log inputs
        if self.log_keys is not None:
            for key in self.log_keys:
                if key in batch:
                    if N > 0:
                        logs[key] = batch[key][:N]
                    else:
                        logs[key] = batch[key]

        if self.global_rank == 0:
            logging.info(f"END log_samples")
        return logs
