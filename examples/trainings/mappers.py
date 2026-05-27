import logging
from typing import List, Optional

from nano_t2i.data.filters import KeyFilter, KeyFilterConfig
from nano_t2i.data.mappers import (
    DatasetCaptionsConfig,
    GetCaptionFromJsonBasedOnNameConfig,
    GetCaptionFromJsonBasedOnNameMapper,
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    SelectKeysMapper,
    SelectKeysMapperConfig,
    SetValueConfig,
    SetValueMapper,
    SqueezeMapper,
    SqueezeMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def generic_mappers(
    dataset_name: str = "rosemary",
    captioners_names: List[str] = ["caption_blip2", "caption_cogvlm"],
    captioners_probabilities: List[float] = [0.5, 0.5],
    image_key: str = "jpg",
    text_key: str = "json",
    vae_embedding_key: str = "embedding_vae-dc-sana1p5-1p6b-1024px-tiling-128-resolution-512x512.pth",
    text_embedding_key: Optional[str] = None,
    verbose: bool = True,
    handles_image: bool = False,
):

    keys_to_select = [text_key, vae_embedding_key, "__url__"]
    key_map = {
        vae_embedding_key: "latent",
    }

    if text_embedding_key is not None:
        keys_to_select.append(text_embedding_key)
        key_map[text_embedding_key] = "text_embedding"
    if handles_image:
        keys_to_select.append(image_key)
        key_map[image_key] = "image"

    mappers = [
        KeyFilter(
            KeyFilterConfig(
                keys=keys_to_select,
                verbose=verbose,
            )
        ),
        SelectKeysMapper(
            SelectKeysMapperConfig(
                keys=keys_to_select,
            )
        ),
    ]

    mappers.extend(
        [
            MapperWrapper(
                [
                    SetValueMapper(
                        SetValueConfig(
                            key="dataset_name",
                            value=dataset_name,
                        )
                    ),
                    GetCaptionFromJsonBasedOnNameMapper(
                        GetCaptionFromJsonBasedOnNameConfig(
                            dataset_name_key="dataset_name",
                            json_key=text_key,
                            output_key="text",
                            configs={
                                dataset_name: DatasetCaptionsConfig(
                                    caption_keys=captioners_names,
                                    caption_probabilities=captioners_probabilities,
                                ),
                            },
                        )
                    ),
                    KeyRenameMapper(
                        KeyRenameMapperConfig(
                            key_map=key_map,
                            verbose=verbose,
                        )
                    ),
                    SqueezeMapper(
                        SqueezeMapperConfig(
                            key="latent",
                            output_key="latent",
                            dim=0,
                        )
                    ),
                    SqueezeMapper(
                        SqueezeMapperConfig(
                            key="latent",
                            output_key="latent",
                            dim=0,
                        )
                    ),
                ],
            )
        ]
    )
    if text_embedding_key is not None:
        mappers.extend(
            [
                SqueezeMapper(
                    SqueezeMapperConfig(
                        key="text_embedding",
                        output_key="text_embedding",
                        dim=0,
                    )
                )
            ]
        )
    if handles_image:
        mappers.extend(
            [
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="image",
                        transforms=["ToTensor"],
                        transforms_kwargs=[{}],
                    )
                )
            ]
        )
    return mappers
