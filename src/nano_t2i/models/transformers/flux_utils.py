# Adapted from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3

    FLASH_ATTN_V3_AVAILABLE = True
except ImportError:
    logging.warning(
        "flash_attn_interface not found, using torch.nn.functional.scaled_dot_product_attention"
    )
    FLASH_ATTN_V3_AVAILABLE = False


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum(
        "...n,d->...nd", pos, omega
    )  # pos.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention_flash_attn_v3(
    q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attention_mask: Tensor = None
) -> Tensor:
    q, k = apply_rope(q, k, pe)
    q = q.transpose(1, 2).to(torch.bfloat16)
    k = k.transpose(1, 2).to(torch.bfloat16)
    v = v.transpose(1, 2).to(torch.bfloat16)
    x = flash_attn_func_v3(q, k, v)
    x = rearrange(x, "B L H D -> B L (H D)")
    return x


def attention(
    q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attention_mask: Tensor = None
) -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attention_mask
    )
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        linear_attention: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        attention_mask: Tensor = None,
    ) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        if FLASH_ATTN_V3_AVAILABLE:
            x = attention_flash_attn_v3(q, k, v, pe=pe, attention_mask=attention_mask)
        else:
            x = attention(q, k, v, pe=pe, attention_mask=attention_mask)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)
        self.dim = dim

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

    @torch.no_grad()
    def adaln_zero_init(self):
        w = self.lin.weight.clone()
        b = self.lin.bias.clone()
        for gate_idx in range(2, self.multiplier, 3):
            out_slice = slice(gate_idx * self.dim, (gate_idx + 1) * self.dim)
            w[out_slice].zero_()
            b[out_slice].zero_()
        self.lin.weight.copy_(w)
        self.lin.bias.copy_(b)


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        use_adaln_learnable_embedding=False,
        txt_modulation_layer: Modulation | None = None,
        img_modulation_layer: Modulation | None = None,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = img_modulation_layer or Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = txt_modulation_layer or Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.use_adaln_learnable_embedding = use_adaln_learnable_embedding

        if use_adaln_learnable_embedding:
            self.adaln_learnable_img_embedding_shift_1 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_learnable_img_embedding_scale_1 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_learnable_img_embedding_gate_1 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_learnable_txt_embedding_shift_1 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_learnable_txt_embedding_scale_1 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_learnable_txt_embedding_gate_1 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_learnable_img_embedding_shift_2 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_learnable_img_embedding_scale_2 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_learnable_img_embedding_gate_2 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_learnable_txt_embedding_shift_2 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_learnable_txt_embedding_scale_2 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_learnable_txt_embedding_gate_2 = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )

    @torch.no_grad()
    def adaln_zero_init(self):
        self.img_mod.adaln_zero_init()
        self.txt_mod.adaln_zero_init()

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        attention_mask: Tensor = None,
        attention_type: str = "flash_attn",
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        if self.use_adaln_learnable_embedding:
            img_mod1.shift = img_mod1.shift * self.adaln_learnable_img_embedding_shift_1
            img_mod1.scale = img_mod1.scale * self.adaln_learnable_img_embedding_scale_1
            img_mod1.gate = img_mod1.gate * self.adaln_learnable_img_embedding_gate_1
            txt_mod1.shift = txt_mod1.shift * self.adaln_learnable_txt_embedding_shift_1
            txt_mod1.scale = txt_mod1.scale * self.adaln_learnable_txt_embedding_scale_1
            txt_mod1.gate = txt_mod1.gate * self.adaln_learnable_txt_embedding_gate_1
            img_mod2.shift = img_mod2.shift * self.adaln_learnable_img_embedding_shift_2
            img_mod2.scale = img_mod2.scale * self.adaln_learnable_img_embedding_scale_2
            img_mod2.gate = img_mod2.gate * self.adaln_learnable_img_embedding_gate_2
            txt_mod2.shift = txt_mod2.shift * self.adaln_learnable_txt_embedding_shift_2
            txt_mod2.scale = txt_mod2.scale * self.adaln_learnable_txt_embedding_scale_2
            txt_mod2.gate = txt_mod2.gate * self.adaln_learnable_txt_embedding_gate_2

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        if FLASH_ATTN_V3_AVAILABLE:
            attn = attention_flash_attn_v3(
                q, k, v, pe=pe, attention_mask=attention_mask
            )
        else:
            attn = attention(q, k, v, pe=pe, attention_mask=attention_mask)

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        linear_attention: bool = False,
        use_adaln_learnable_embedding=False,
        modulation_layer: Modulation | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = modulation_layer or Modulation(hidden_size, double=False)
        self.use_adaln_learnable_embedding = use_adaln_learnable_embedding
        if use_adaln_learnable_embedding:
            self.adaln_per_block_embedding_shift = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_per_block_embedding_scale = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )
            self.adaln_per_block_embedding_gate = nn.Parameter(
                torch.randn(hidden_size) * 0.02
            )

    @torch.no_grad()
    def adaln_zero_init(self):
        self.modulation.adaln_zero_init()

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        attention_mask: Tensor = None,
        attention_type: str = "flash_attn",
    ) -> Tensor:
        mod, _ = self.modulation(vec)

        if self.use_adaln_learnable_embedding:
            mod.shift = mod.shift * self.adaln_per_block_embedding_shift
            mod.scale = mod.scale * self.adaln_per_block_embedding_scale
            mod.gate = mod.gate * self.adaln_per_block_embedding_gate

        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        if FLASH_ATTN_V3_AVAILABLE:
            attn = attention_flash_attn_v3(
                q, k, v, pe=pe, attention_mask=attention_mask
            )
        else:
            attn = attention(q, k, v, pe=pe, attention_mask=attention_mask)

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
