from .mappers_batched_config import BaseMapperBatchedConfig


class BaseMapperBatched:
    """
    Base class for the mappers used to modify the samples in the data pipeline.

    Args:

        config (BaseMapperBatchedConfig):
            Configuration for the mapper.
    """

    def __init__(self, config: BaseMapperBatchedConfig):
        self.config = config
        self.key = config.key

        if config.output_key is None:
            self.output_key = config.key
        else:
            self.output_key = config.output_key

    def map(self):
        raise NotImplementedError
