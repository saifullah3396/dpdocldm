from dataclasses import dataclass
from typing import List, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from atria.models.autoencoding.modules import Decoder, Encoder
from diffusers.models.autoencoders.autoencoder_kl import (
    AutoencoderKLOutput,
    DecoderOutput,
)
from taming.modules.vqvae.quantize import VectorQuantizer
from torch import nn


@dataclass
class AutoencoderVQOutput:
    quant: torch.FloatTensor
    emb_loss: torch.FloatTensor
    info: torch.FloatTensor


class VQModel(pl.LightningModule):
    def __init__(
        self,
        n_embed: int = 8192,
        model_channels: int = 128,
        out_channels: int = 3,
        double_z: bool = True,
        channel_mults: List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        attn_resolutions: List[int] = [],
        dropout: float = 0.0,
        in_channels: int = 3,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
        resamp_with_conv: bool = True,
        z_channels: int = 3,
        embed_dim: int = (
            3  # -> this turns the image final dimension to h x w x embed_dim
        ),
        image_size: int = 256,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        self._n_embed = n_embed
        self._model_channels = model_channels
        self._out_channels = out_channels
        self._double_z = double_z
        self._channel_mults = channel_mults
        self._num_res_blocks = num_res_blocks
        self._attn_resolutions = attn_resolutions
        self._dropout = dropout
        self._in_channels = in_channels
        self._use_linear_attn = use_linear_attn
        self._attn_type = attn_type
        self._resamp_with_conv = resamp_with_conv
        self._z_channels = z_channels
        self._embed_dim = embed_dim
        self._image_size = image_size
        self._remap = remap
        self._sane_index_shape = sane_index_shape
        self._scaling_factor = scaling_factor

        self._build_model()

    @property
    def scaling_factor(self):
        return self._scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, value):
        self._scaling_factor = value

    def _build_enc_dec(self, encoder: bool = True):
        model_class = Encoder if encoder else Decoder
        return model_class(
            ch=self._model_channels,
            out_ch=self._out_channels,
            ch_mult=self._channel_mults,
            num_res_blocks=self._num_res_blocks,
            attn_resolutions=self._attn_resolutions,
            dropout=self._dropout,
            in_channels=self._in_channels,
            resamp_with_conv=self._resamp_with_conv,
            resolution=self._image_size,
            z_channels=self._z_channels,
            double_z=self._double_z,
            use_linear_attn=self._use_linear_attn,
            attn_type=self._attn_type,
        )

    def _build_model(self):
        # pass init params to Encoder
        self.encoder = self._build_enc_dec(encoder=True)

        # pass init params to Decoder
        self.decoder = self._build_enc_dec(encoder=False)

        self.quantize = VectorQuantizer(
            self._n_embed,
            self._embed_dim,
            beta=0.25,
            remap=self._remap,
            sane_index_shape=self._sane_index_shape,
        )
        self.quant_conv = nn.Conv2d(self._z_channels, self._embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(self._embed_dim, self._z_channels, 1)

        self.use_slicing = False
        self.use_tiling = False

    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> AutoencoderKLOutput:
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)

        if not return_dict:
            return quant, emb_loss, info

        return AutoencoderVQOutput(quant=quant, emb_loss=emb_loss, info=info)

    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(
        self,
        sample: torch.FloatTensor,
        return_pred_indices: bool = False,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        quant, diff, (_, _, ind) = self.encode(sample)
        dec = self.decode(quant)

        if not return_dict:
            if return_pred_indices:
                return dec, diff, ind
            return dec, diff

        return DecoderOutput(sample=dec, diff=diff, indices=ind)
