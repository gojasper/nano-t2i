import itertools
import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import Resize

from ..datasets.collation_fn import custom_collation_fn
from .base import BaseMapperBatched
from .mappers_batched_config import MultiAspectRatioCacherConfig


@dataclass
class BucketsWithMaxBudget:
    pixel_budget: int
    aspect_ratio: float
    interval: int
    buckets: List[Tuple[int, int, float]]


class MultiAspectRatioCacherMapper(BaseMapperBatched):
    """MultiAspectRatioCacherMapper

    This mapper caches input images (`image`, `mask`, `masked_image`) into buckets by
        - Resizing them to target pixel budgets
        - Resizing them to target aspect ratios for this given pixel budget

    Each bucket is defined by a probability, pixel budget, aspect ratio, round_interval and batch size.
    Since the size of the image is modified, the mapper computed on the fly the original-size, crop-coordinates and target-size for each bucket and stores them in the batch.

    Warning: the mapper expects batches of torch.Tensor images of length 1, make sure to use per_worker_batch_size=1

    Args:
        config (MultiAspectRatioCacherConfig): Configuration for the mapper
    """

    def __init__(self, config: MultiAspectRatioCacherConfig):
        super().__init__(config)
        # define parameters
        self.pixel_budgets = config.pixel_budgets
        self.aspect_ratios = config.aspect_ratios
        self.intervals = config.intervals
        self.batch_sizes = config.batch_sizes
        self.precision = config.precision

        self.probabilites = [
            prob * batch_size / sum(config.batch_sizes)
            for prob, batch_size in zip(config.probabilities, config.batch_sizes)
        ]
        self.probabilites = [
            prob / sum(self.probabilites) for prob in self.probabilites
        ]

        # set buckets for each pixel budget
        self.all_buckets: List[BucketsWithMaxBudget] = []
        for pixel_budget, aspect_ratio, interval in zip(
            self.pixel_budgets, self.aspect_ratios, self.intervals
        ):
            buckets_with_max_budget = self._get_buckets(
                pixel_budget=pixel_budget,
                aspect_ratio=aspect_ratio,
                interval=interval,
                precision=self.precision,
            )
            self.all_buckets.append(buckets_with_max_budget)
            logging.debug(f"Created bucket: {buckets_with_max_budget}")
        self.buckets_ids = list(range(0, len(self.all_buckets)))
        self.buckets_ids_pixel_budget = {
            k: v for k, v in zip(self.pixel_budgets, self.buckets_ids)
        }

        # set cache for each bucket
        self.bucket_cache = []
        for bucket in self.all_buckets:
            self.bucket_cache.append({(w, h): [] for w, h, _ in bucket.buckets})

        # set input keys
        self.input_keys = config.input_keys
        self.output_keys = config.output_keys

        # original, crop, target keys
        self.original_key = config.original_key
        self.crop_key = config.crop_key
        self.target_key = config.target_key

        # batched transforms
        self.filters_mappers = config.filters_mappers

        self.thumbnail_strategies = config.thumbnail_strategies
        self.thumbnail_strategies_probabilities = (
            config.thumbnail_strategies_probabilities
        )

    def __call__(
        self,
        batch: dict,
        pixel_budgets: Optional[List[int]] = None,
        pixel_budgets_probabilities: Optional[List[float]] = None,
    ) -> Union[dict, None]:
        """
        Call the mapper that will resize the input images to the closest bucket size and aspect ratio. If probabilities and pixel budgets are provided, the mapper will sample a bucket based on the probabilities and pixel budgets. The pixel budgets must be a subset of the original pixel budgets and the probabilities must be sum to 1.

        Args:
            batch (dict): batch to map
            pixel_budgets (Optional[List[int]]): pixel budgets of the buckets
            pixel_budgets_probabilities (Optional[List[float]]): probabilities of the pixel budgets
        Returns:
        """

        if pixel_budgets_probabilities is not None or pixel_budgets is not None:
            assert (
                pixel_budgets_probabilities is not None and pixel_budgets is not None
            ), "pixel_budgets_probabilities and pixel_budgets must be provided"
            assert len(pixel_budgets_probabilities) == len(
                pixel_budgets
            ), "pixel_budgets_probabilities and pixel_budgets must have the same length"
            assert set(pixel_budgets).issubset(
                set(self.pixel_budgets)
            ), f"pixel_budgets must be a subset of the original pixel_budgets: {pixel_budgets} vs {self.pixel_budgets}"

            assert (
                sum(pixel_budgets_probabilities) == 1
            ), "pixel_budgets_probabilities must be sum to 1"
            buckets_ids = [
                self.buckets_ids_pixel_budget[pixel_budget]
                for pixel_budget in pixel_budgets
            ]

        else:
            pixel_budgets_probabilities = self.probabilites
            pixel_budgets = self.pixel_budgets
            buckets_ids = self.buckets_ids

        # make sure that all batch items are of length 1
        for batch_item in batch:
            assert (
                len(batch[batch_item]) == 1
            ), f"{batch_item} is not a list of length 1, {batch[batch_item].shape}"
            batch[batch_item] = batch[batch_item][0]

        # make sure that inputs are torch tensors
        for input_key in self.input_keys:
            assert isinstance(
                batch[input_key], torch.Tensor
            ), f"{input_key} is not a torch tensor"
            assert len(batch[input_key].size()) == 3, f"{input_key} is not a 3D tensor"
            assert (
                batch[input_key].min() >= 0 and batch[input_key].max() <= 1
            ), "images should be in [0, 1] range"

        # when multiple input keys, we need to make sure that they are all of the same size
        if len(self.input_keys) > 1:
            for input_key in self.input_keys:
                if batch[input_key].shape[1:] != batch[self.input_keys[0]].shape[1:]:
                    batch[input_key] = Resize(
                        size=batch[self.input_keys[0]].shape[1:],
                    )(batch[input_key])
                assert (
                    batch[input_key].shape[1:] == batch[self.input_keys[0]].shape[1:]
                ), (
                    f"images {input_key} and {self.input_keys[0]} are not of the same size: "
                    f"{batch[input_key].shape[1:]} vs {batch[self.input_keys[0]].shape[1:]}"
                )

        # make a copy of the batch
        sample = batch.copy()

        bucket_list_index = random.choices(
            buckets_ids, weights=pixel_budgets_probabilities
        )[0]
        selected_bucket = self.all_buckets[bucket_list_index]
        bucket_cache = self.bucket_cache[bucket_list_index]
        batch_size = self.batch_sizes[bucket_list_index]

        for i, (input_key, output_key) in enumerate(
            zip(self.input_keys, self.output_keys)
        ):
            image_input = batch[input_key]

            # torch to pil for easier manipulation
            image_input = transforms.ToPILImage()(image_input)
            width, height = image_input.size
            width_original, height_original = width, height
            image_aspect_ratio = width / height
            logging.debug(
                f"Original size: {width}x{height}. Aspect ratio: {image_aspect_ratio}"
            )

            # get the closest bucket based on the aspect ratio
            width_target, height_target, bucket_aspect_ratio = min(
                selected_bucket.buckets, key=lambda x: abs(x[2] - image_aspect_ratio)
            )
            logging.debug(
                f"Closest bucket: {width_target}x{height_target}. Bucket aspect ratio: {bucket_aspect_ratio}"
            )

            # creates a thumbnail of the input image with the specified width and height
            thumbnail_strategy = random.choices(
                self.thumbnail_strategies,
                weights=self.thumbnail_strategies_probabilities,
            )[0]

            image_input.thumbnail(
                size=(width_target, height_target),
                resample=Image.Resampling.LANCZOS,
            )
            width, height = image_input.size

            image_input = ImageOps.fit(
                image_input,
                (width_target, height_target),
                bleed=0.0,
                centering=(0.5, 0.5),
                method=Image.Resampling.LANCZOS,
            )

            image_input = transforms.ToTensor()(image_input)

            sample[output_key] = image_input

            if i == 0:
                if self.original_key:
                    sample[self.original_key] = torch.Tensor(
                        [width_original, height_original]
                    ).unsqueeze(0)

                if self.crop_key:
                    crop_width = (width_target - width) / 2
                    crop_height = (height_target - height) / 2
                    sample[self.crop_key] = torch.Tensor(
                        [crop_width, crop_height]
                    ).unsqueeze(0)

                if self.target_key:
                    sample[self.target_key] = torch.Tensor(
                        [width_target, height_target]
                    ).unsqueeze(0)

        # check if the cache bucket was already opened, creates one otherwise
        if (width_target, height_target) in bucket_cache:
            bucket_cache[(width_target, height_target)].append(sample)
        else:
            bucket_cache[(width_target, height_target)] = [sample]

        # if the cache size == batch size, returns the batch
        if len(bucket_cache[(width_target, height_target)]) == batch_size:
            # batch the examples
            batch = custom_collation_fn(
                bucket_cache[(width_target, height_target)]
            ).copy()

            # apply the batched transforms
            for mapper in self.filters_mappers:
                batch = mapper(batch)

            # clean the cache
            del bucket_cache[(width_target, height_target)]
            return batch
        # if the batch is not full yet, return None
        else:
            return None

    def _round_to_interval(self, size: float, interval_round: int) -> int:
        """Round to the nearest interval_rond

        Args:
            size (int): size to round
            interval_round (int): interval to round to

        Returns:
            int: rounded size
        """
        return round(int(size) / interval_round) * interval_round

    def _get_buckets(
        self,
        pixel_budget: int,
        aspect_ratio: float,
        interval: int,
        precision: float = 0.9,
    ) -> BucketsWithMaxBudget:
        """
        Generate a list of possible sizes for a given pixel budget and aspect ratio.
        This method calculates possible width and height combinations that fit within
        the specified pixel budget and aspect ratio, rounded to the nearest interval.

        Args:
            pixel_budget (int): Maximum pixel budget.
            aspect_ratio (float): Maximum aspect ratio.
            interval (int): Interval to round to.
            precision (float, optional): Precision to be close to pixel_budget. Defaults to 0.9.

        Returns:
            BucketsWithMaxBudget

        Raises:
            ValueError: If no valid buckets are found.
        """
        assert aspect_ratio >= 1, "Aspect ratio should be >= 1"

        # Calculate minimum and maximum edge lengths
        min_edge = self._round_to_interval(
            (pixel_budget / aspect_ratio) ** 0.5, interval_round=interval
        )
        max_edge = self._round_to_interval(
            (pixel_budget * aspect_ratio) ** 0.5, interval_round=interval
        )

        # Generate possible sizes within the calculated range
        possible_sizes = range(min_edge, max_edge + 1, interval)

        # Create buckets using cartesian product of possible sizes
        buckets = [
            (width, height)
            for width, height in itertools.product(possible_sizes, repeat=2)
            if width * height <= pixel_budget
        ]

        # Filter buckets based on precision
        buckets = [
            (width, height)
            for width, height in buckets
            if width * height >= (pixel_budget * precision)
        ]

        # Add aspect ratio to each bucket and round to 1e-2
        buckets = [
            (width, height, round(width / height, 2)) for width, height in buckets
        ]

        # Sort buckets by aspect ratio
        buckets.sort(key=lambda x: x[2])

        if not buckets:
            raise ValueError(
                f"No valid buckets found for pixel budget {pixel_budget} and aspect ratio {aspect_ratio}. "
                f"Consider decreasing the precision or the interval."
            )

        return BucketsWithMaxBudget(
            buckets=buckets,
            pixel_budget=pixel_budget,
            aspect_ratio=aspect_ratio,
            interval=interval,
        )
