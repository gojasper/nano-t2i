from dataclasses import field
from typing import List, Literal, Optional, Union

from pydantic.dataclasses import dataclass

from ..config import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """
    Configuration for the training pipeline

    Args:

        experiment_id (str):
            The experiment id for the training run. If not provided, a random id will be generated.
        optimizers_name (List[str]):
            The list of optimizers to use. Default is ["AdamW"]. Choices are "Adam", "AdamW", "Adadelta", "Adagrad", "RMSprop", "SGD"
        optimizers_num_backward_steps (List[int]):
            The list of consecutive global backward steps per optimizer (used with multiple optimizers). Default is None.
            If None, optimizers_num_backward_steps is set to a list of 1s of length equal to the number of optimizers
        optimizers_kwargs (List[Dict[str, Any]])
            The optimizers kwargs. Default is [{}]
        learning_rates (List[float]):
            The learning rates to use for each optimizer. Default is [1e-3]
        lr_schedulers_name (List[str]):
            The learning rate schedulers to use. Default is [None]. Choices are "StepLR", "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "ExponentialLR"
        lr_schedulers_kwargs (List[Dict[str, Any]])
            The learning rate schedulers kwargs. Default is [{}]
        lr_schedulers_interval (List[str]):
            The learning rate scheduler intervals. Default is ["step"]. Choices are "step", "epoch"
        lr_schedulers_frequency (List[int]):
            The learning rate scheduler frequency. Default is 1
        metrics (List[str])
            The metrics to use. Default is None
        tracking_metrics: Optional[List[str]]
            The metrics to track. Default is None
        backup_every (int):
            The frequency to backup the model. Default is 50.
        trainable_params (Union[List[str], List[List[str]]]):
            Regexes indicateing the parameters to train for each optimizer.
            Default is [["./*"]] (i.e. all parameters are trainable)
        log_keys: Union[str, List[str]]:
            The keys to log when sampling from the model. Default is "txt"
        log_samples_model_kwargs (Dict[str, Any]):
            The kwargs for logging samples from the model. Default is {
                "max_samples": 8,
                "num_steps": 20,
                "input_shape": (4, 32, 32),
                "guidance_scale": 7.5,
            }
        accumulate_grad_batches (int):
            Gradient accumulation parameter for training stage. Default is 1.
        limit_val_batches (int):
            Accumulation parameter for validation stage. Default is 1.
            It can be set to 0 if validation is disabled.
    """

    experiment_id: Optional[str] = None
    optimizers_name: List[
        Literal["Adam", "AdamW", "Adadelta", "Adagrad", "RMSprop", "SGD"]
    ] = field(default_factory=lambda: ["AdamW"])
    optimizers_num_backward_steps: Optional[List[int]] = None
    optimizers_kwargs: Optional[Union[List[dict]]] = field(default_factory=lambda: [{}])
    learning_rates: List[float] = field(default_factory=lambda: [1e-3])
    lr_schedulers_name: Optional[
        List[
            Literal[
                "StepLR",
                "CosineAnnealingLR",
                "CosineAnnealingWarmRestarts",
                "ReduceLROnPlateau",
                "ExponentialLR",
                None,
            ]
        ]
    ] = field(default_factory=lambda: [None])
    lr_schedulers_kwargs: Optional[List[dict]] = field(default_factory=lambda: [{}])
    lr_schedulers_interval: Optional[List[Literal["step", "epoch", None]]] = field(
        default_factory=lambda: ["step"]
    )
    lr_schedulers_frequency: Optional[List[Union[int, None]]] = field(
        default_factory=lambda: [1]
    )
    metrics: Optional[List[str]] = None
    tracking_metrics: Optional[List[str]] = None
    backup_every: int = 50
    trainable_params: Optional[List[List[str]]] = field(
        default_factory=lambda: [["./*"]]
    )
    log_keys: Optional[Union[str, List[str]]] = "txt"
    log_samples_model_kwargs: Optional[dict] = field(
        default_factory=lambda: {
            "max_samples": 8,
            "num_steps": 20,
            "input_shape": (4, 32, 32),
            "guidance_scale": 7.5,
        }
    )
    accumulate_grad_batches: int = 1
    limit_val_batches: int = 1
    optimizer1_start_step: int = 0
    optimizer1_warmup_steps: Optional[int] = None
    optimizer1_rewarm_steps: Optional[int] = None
    optimizer1_rewarm_num_steps: Optional[int] = None

    def __post_init__(self):
        # only supports one optimizer for now
        assert len(self.optimizers_name) == 1, "Only one optimizer is supported for now"
        assert (
            len(self.learning_rates) == 1
        ), "Only one learning rate is supported for now"

        self.optimizers_kwargs = [self.optimizers_kwargs[0]]
        self.trainable_params = [self.trainable_params[0]]

        # if lr_scheduler_kwargs provided check len
        if self.lr_schedulers_kwargs != [{}]:
            assert len(self.lr_schedulers_name) == len(
                self.lr_schedulers_kwargs
            ), f"The length of lr_schedulers_name ({len(self.lr_schedulers_name)}) must be equal to the length of lr_schedulers_kwargs ({len(self.lr_schedulers_kwargs)})"
            if self.lr_schedulers_frequency != [1]:
                assert len(self.lr_schedulers_name) == len(
                    self.lr_schedulers_frequency
                ), f"The length of lr_schedulers_name ({len(self.lr_schedulers_name)}) must be equal to the length of lr_schedulers_frequency ({len(self.lr_schedulers_frequency)})"
            else:
                self.lr_schedulers_frequency = [1 for _ in self.lr_schedulers_name]

            if self.lr_schedulers_interval != ["step"]:
                assert len(self.lr_schedulers_name) == len(
                    self.lr_schedulers_interval
                ), f"The length of lr_schedulers_name ({len(self.lr_schedulers_name)}) must be equal to the length of lr_schedulers_interval ({len(self.lr_schedulers_interval)})"
            else:
                self.lr_schedulers_interval = ["step" for _ in self.lr_schedulers_name]

        else:
            self.lr_schedulers_kwargs = [{} for _ in self.lr_schedulers_name]

        assert len(self.optimizers_name) == len(
            self.learning_rates
        ), f"The length of optimizers_name ({len(self.optimizers_name)}) must be equal to the length of learning_rates ({len(self.learning_rates)})"

        if len(self.optimizers_name) > 1:
            self.optimizers_num_backward_steps = (
                [1] * len(self.optimizers_name)
                if self.optimizers_num_backward_steps is None
                else self.optimizers_num_backward_steps
            )

            assert len(self.optimizers_name) == len(
                self.optimizers_num_backward_steps
            ), f"The length of optimizers_name ({len(self.optimizers_name)}) must be equal to the length of optimizers_num_backward_steps ({len(self.optimizers_num_backward_steps)})"

            assert sum(
                [num_steps > 0 for num_steps in self.optimizers_num_backward_steps]
            ) == len(
                self.optimizers_num_backward_steps
            ), "Each optimizer should at least be updated once."

        assert (
            self.accumulate_grad_batches > 0
        ), "Gradient accumulation must be at least greater than 1 (default parameter)."

        if self.optimizer1_warmup_steps is not None:
            assert (
                self.optimizer1_warmup_steps > 0
            ), "Optimizer 1 warmup steps must be at least greater than 0."
        else:
            self.optimizer1_warmup_steps = 0

        if self.optimizer1_rewarm_steps is not None:
            assert (
                self.optimizer1_rewarm_steps > 0
            ), "Optimizer 1 rewarm steps must be at least greater than 0."
        else:
            self.optimizer1_rewarm_steps = 0

        if self.optimizer1_rewarm_num_steps is not None:
            assert (
                self.optimizer1_rewarm_num_steps > 0
            ), "Optimizer 1 rewarm num steps must be at least greater than 0."
        else:
            self.optimizer1_rewarm_num_steps = 0

        super().__post_init__()
