import torch
from diffusers.models import AutoencoderDC

from ..base.base_model import BaseModel
from ..utils import Tiler, pad
from .autoencoderDC_config import AutoencoderDCDiffusersConfig


class AutoencoderDCDiffusers(BaseModel):
    """This is the VAE class used to work with latent models

    Args:

        config (AutoencoderKLDiffusersConfig): The config class which defines all the required parameters.
    """

    def __init__(self, config: AutoencoderDCDiffusersConfig):
        BaseModel.__init__(self, config)
        self.config = config
        if config.version is not None:
            self.vae_model = AutoencoderDC.from_pretrained(
                config.version,
                subfolder=config.subfolder,
                revision=config.revision,
            )
        else:
            print(config.config)
            self.vae_model = AutoencoderDC(
                **config.config,
            )
        self.tiling_size = config.tiling_size
        self.tiling_overlap = config.tiling_overlap
        self.dummy_encoder = config.dummy_encoder

        # get downsampling factor
        self._get_properties()

    def get_decoder_last_layer(self):
        return self.vae_model.decoder.conv_out.weight

    def get_encoder_last_layer(self):
        return self.vae_model.encoder.conv_out.weight

    @torch.no_grad()
    def _get_properties(self):

        # set latent channels
        self.latent_channels = self.vae_model.config.latent_channels
        self.scale_factor = self.vae_model.config.scaling_factor

        x = torch.randn(1, self.vae_model.config.in_channels, 32, 32)
        z = self.encode(x)

        # set downsampling factor
        self.downsampling_factor = int(x.shape[2] / z.shape[2])

    def forward(self, x: torch.tensor, **kwargs):
        return self.decode(self.encode(x, **kwargs))

    def encode(self, x: torch.tensor, batch_size: int = 8, **kwargs):
        if self.dummy_encoder:
            return x
        latents = []
        for i in range(0, x.shape[0], batch_size):
            latents.append(self.vae_model.encode(x[i : i + batch_size]).latent)
        latents = torch.cat(latents, dim=0)
        latents = latents * self.scale_factor

        return latents

    def decode(self, z: torch.tensor, **kwargs):
        use_tiling = (
            z.shape[2] > self.tiling_size[0] or z.shape[3] > self.tiling_size[1]
        )

        z = z / self.scale_factor

        if use_tiling:
            samples = []
            for i in range(z.shape[0]):

                z_i = z[i].unsqueeze(0)

                tiler = Tiler()
                tiles = tiler.get_tiles(
                    input=z_i,
                    tile_size=self.tiling_size,
                    overlap_size=self.tiling_overlap,
                    scale=self.downsampling_factor,
                    out_channels=3,
                )

                for i, tile_row in enumerate(tiles):
                    for j, tile in enumerate(tile_row):
                        tile_shape = tile.shape
                        # pad tile to inference size if tile is smaller than inference size
                        tile = pad(
                            tile,
                            base_h=self.tiling_size[0],
                            base_w=self.tiling_size[1],
                        )
                        tile_decoded = self.vae_model.decode(tile).sample
                        tiles[i][j] = (
                            tile_decoded[
                                0,
                                :,
                                : int(tile_shape[2] * self.downsampling_factor),
                                : int(tile_shape[3] * self.downsampling_factor),
                            ]
                            .cpu()
                            .unsqueeze(0)
                        )

                # merge tiles
                samples.append(tiler.merge_tiles(tiles=tiles))

            samples = torch.cat(samples, dim=0)

        else:
            samples = self.vae_model.decode(z).sample

        return samples
