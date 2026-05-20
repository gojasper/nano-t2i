from dataclasses import field
from typing import List, Literal, Optional

from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class BaseMapperBatchedConfig(BaseConfig):
    """
    Base configuration for batched mappers.

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
class MultiAspectRatioCacherConfig(BaseMapperBatchedConfig):
    """
    Cache input images (`image`, `mask`, `masked_image`) into buckets with different aspect ratios and pixel budgets.
    Each bucket is defined by a probability, pixel budget, aspect ratio, round_interval and batch size.
    The mapper computed on the fly the crop coordinates and target size for each bucket and stores them in the batch.

    Args:

        input_keys (List[str]): Key to apply the mapper to. Defaults to "image"
        output_keys (List[str]): Key to store the output of the mapper. Defaults to "image"
        probabilities (List[float]): List of probabilities for each aspect ratio. Defaults to [1.0]
        pixel_budgets (List[int]): List of pixel budgets for each aspect ratio. Defaults to [1024**2]
        aspect_ratios (List[float]): List of aspect ratios. Defaults to [1.0]
        intervals (List[int]): List of intervals for each aspect ratio. Defaults to [128]
        batch_sizes (List[int]): List of batch sizes for each aspect ratio. Defaults to [1]
        original_key (str): Key to store the original size as a tuple. Defaults to "original_size_as_tuple"
        crop_key (str): Key to store the crop coordinates top left as a tuple. Defaults to "crop_coords_top_left"
        target_key (str): Key to store the target size as a tuple. Defaults to "target_size_as_tuple"
        filters_mappers (List[BaseMapper]): List of mappers to apply to the cached images. Defaults to None
        precision (float): Precision to be close to pixel_budget. Defaults to 0.9
    """

    input_keys: List[str] = field(default_factory=lambda: ["image"])
    output_keys: List[str] = field(default_factory=lambda: ["image"])
    probabilities: Optional[List[float]] = None
    pixel_budgets: List[int] = field(default_factory=lambda: [1024**2])
    aspect_ratios: List[float] = field(default_factory=lambda: [1.0])
    intervals: List[int] = field(default_factory=lambda: [128])
    batch_sizes: List[int] = field(default_factory=lambda: [1])
    original_key: Optional[str] = "original_size_as_tuple"
    crop_key: Optional[str] = "crop_coords_top_left"
    target_key: Optional[str] = "target_size_as_tuple"
    filters_mappers: Optional[list] = None
    precision: float = 0.9
    thumbnail_strategies: List[Literal["crop", "resize"]] = field(
        default_factory=lambda: ["resize"]
    )
    thumbnail_strategies_probabilities: List[float] = field(
        default_factory=lambda: [1.0]
    )

    def __post_init__(self):
        super().__post_init__()
        assert (
            len(self.pixel_budgets)
            == len(self.aspect_ratios)
            == len(self.intervals)
            == len(self.batch_sizes)
        ), "All lists must have the same length"
        if self.probabilities is not None:
            assert len(self.probabilities) == len(
                self.pixel_budgets
            ), "Probabilities must have the same length as pixel_budgets"
        else:
            self.probabilities = [1.0 / len(self.pixel_budgets)] * len(
                self.pixel_budgets
            )

        assert sum(self.probabilities) == 1, "Probabilities must sum to 1"

        if self.filters_mappers is None:
            self.filters_mappers = []
