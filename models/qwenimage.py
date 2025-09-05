
# -*- coding: utf-8 -*-
"""
Nunchaku Qwen-Image (patched) — ControlNet working with CPU offload

- Keeps InstantX / Union ControlNet support (per-block residual add)
- Keeps patches_replace hooks (DiffSynth etc.)
- Allows CPU offload to stay enabled (no forced disable); makes residual moves non_blocking
- Safe residual add: device/dtype align + token-length overlap

Drop-in replacement for:
custom_nodes/ComfyUI-nunchaku/model_base/qwenimage.py
"""

import gc
from typing import Optional, Tuple, Dict, Any, List

import torch
from torch import nn

from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.qwen_image.model import (
    GELU,
    FeedForward,
    LastLayer,
    QwenImageTransformer2DModel,
    QwenTimestepProjEmbeddings,
    apply_rotary_emb,
)

from nunchaku.models.linear import AWQW4A16Linear, SVDQW4A4Linear
from nunchaku.models.utils import CPUOffloadManager
from nunchaku.ops.fused import fused_gelu_mlp

from ..mixins.model import NunchakuModelMixin


# ---------- helpers ----------
def _to_dev_dtype(t: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    if t is None or not isinstance(t, torch.Tensor):
        return None
    if t.device == ref.device and t.dtype == ref.dtype:
        return t
    # non_blocking=True if source in pinned memory; harmless otherwise
    return t.to(device=ref.device, dtype=ref.dtype, non_blocking=True)


def _apply_residual(base: torch.Tensor, resid: Optional[torch.Tensor], scale: float = 1.0) -> torch.Tensor:
    if resid is None:
        return base
    if resid.numel() == 0:
        return base
    # Try fast broadcast first
    try:
        base.add_(resid, alpha=scale)
        return base
    except Exception:
        pass
    # If sequence lengths differ, add overlap only
    if base.dim() == 3 and resid.dim() == 3 and base.shape[0] == resid.shape[0] and base.shape[2] == resid.shape[2]:
        b, sb, d = base.shape
        _, sr, _ = resid.shape
        s = min(sb, sr)
        if s > 0:
            base[:, :s, :].add_(resid[:, :s, :], alpha=scale)
        return base
    # Fallback: try to expand dims
    tmp = resid
    try:
        while tmp.dim() < base.dim():
            tmp = tmp.unsqueeze(1)
        base.add_(tmp, alpha=scale)
    except Exception:
        pass
    return base


# ---------- quantized layers ----------
class NunchakuGELU(GELU):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, dtype=None, device=None, **kwargs):
        super(GELU, self).__init__()
        self.proj = SVDQW4A4Linear(dim_in, dim_out, bias=bias, torch_dtype=dtype, device=device, **kwargs)
        self.approximate = approximate


class NunchakuFeedForward(FeedForward):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        inner_dim=None,
        bias: bool = True,
        dtype=None,
        device=None,
        **kwargs,
    ):
        super(FeedForward, self).__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([])
        self.net.append(NunchakuGELU(dim, inner_dim, approximate="tanh", bias=bias, dtype=dtype, device=device, **kwargs))
        self.net.append(nn.Dropout(dropout))
        self.net.append(
            SVDQW4A4Linear(
                inner_dim,
                dim_out,
                bias=bias,
                act_unsigned=kwargs.get("precision", None) == "int4",
                torch_dtype=dtype,
                device=device,
                **kwargs,
            )
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.net[0], NunchakuGELU):
            return fused_gelu_mlp(hidden_states, self.net[0].proj, self.net[2])
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-5,
        out_bias: bool = True,
        out_dim: int = None,
        out_context_dim: int = None,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.heads = heads
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.dropout = dropout

        self.norm_q = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

        self.to_qkv = SVDQW4A4Linear(query_dim, self.inner_dim + self.inner_kv_dim * 2, bias=bias, torch_dtype=dtype, device=device, **kwargs)
        self.add_qkv_proj = SVDQW4A4Linear(query_dim, self.inner_dim + self.inner_kv_dim * 2, bias=bias, torch_dtype=dtype, device=device, **kwargs)

        self.to_out = nn.ModuleList([
            SVDQW4A4Linear(self.inner_dim, self.out_dim, bias=out_bias, torch_dtype=dtype, device=device, **kwargs),
            nn.Dropout(dropout),
        ])
        self.to_add_out = SVDQW4A4Linear(self.inner_dim, self.out_context_dim, bias=out_bias, torch_dtype=dtype, device=device, **kwargs)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        seq_txt = encoder_hidden_states.shape[1]

        img_qkv = self.to_qkv(hidden_states)
        img_query, img_key, img_value = img_qkv.chunk(3, dim=-1)

        txt_qkv = self.add_qkv_proj(encoder_hidden_states)
        txt_query, txt_key, txt_value = txt_qkv.chunk(3, dim=-1)

        img_query = img_query.unflatten(-1, (self.heads, -1))
        img_key = img_key.unflatten(-1, (self.heads, -1))
        img_value = img_value.unflatten(-1, (self.heads, -1))

        txt_query = txt_query.unflatten(-1, (self.heads, -1))
        txt_key = txt_key.unflatten(-1, (self.heads, -1))
        txt_value = txt_value.unflatten(-1, (self.heads, -1))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_query = apply_rotary_emb(joint_query, image_rotary_emb)
        joint_key = apply_rotary_emb(joint_key, image_rotary_emb)

        joint_query = joint_query.flatten(start_dim=2)
        joint_key = joint_key.flatten(start_dim=2)
        joint_value = joint_value.flatten(start_dim=2)

        joint_hidden_states = optimized_attention_masked(joint_query, joint_key, joint_value, self.heads, attention_mask)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class NunchakuQwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None,
        scale_shift: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.scale_shift = scale_shift
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(nn.SiLU(), AWQW4A16Linear(dim, 6 * dim, bias=True, torch_dtype=dtype, device=device))
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = NunchakuFeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, **kwargs)

        self.txt_mod = nn.Sequential(nn.SiLU(), AWQW4A16Linear(dim, 6 * dim, bias=True, torch_dtype=dtype, device=device))
        self.txt_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_mlp = NunchakuFeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, **kwargs)

        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            dtype=dtype,
            device=device,
            operations=operations,
            **kwargs,
        )

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if self.scale_shift != 0:
            scale.add_(self.scale_shift)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)

        img_mod_params = img_mod_params.view(img_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(img_mod_params.shape[0], -1)
        txt_mod_params = txt_mod_params.view(txt_mod_params.shape[0], -1, 6).transpose(1, 2).reshape(txt_mod_params.shape[0], -1)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            attention_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return encoder_hidden_states, hidden_states


class NunchakuQwenImageTransformer2DModel(NunchakuModelMixin, QwenImageTransformer2DModel):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        image_model=None,
        dtype=None,
        device=None,
        operations=None,
        scale_shift: float = 1.0,
        **kwargs,
    ):
        super(QwenImageTransformer2DModel, self).__init__()
        self.dtype = dtype
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.txt_norm = operations.RMSNorm(joint_attention_dim, eps=1e-6, dtype=dtype, device=device)
        self.img_in = operations.Linear(in_channels, self.inner_dim, dtype=dtype, device=device)
        self.txt_in = operations.Linear(joint_attention_dim, self.inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [
                NunchakuQwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                    scale_shift=scale_shift,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = LastLayer(self.inner_dim, self.inner_dim, dtype=dtype, device=device, operations=operations)
        self.proj_out = operations.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True, dtype=dtype, device=device)
        self.gradient_checkpointing = False

        self.offload = False
        self.offload_manager: Optional[CPUOffloadManager] = None

    def _forward(
        self,
        x: torch.Tensor,
        timesteps,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        ref_latents=None,
        transformer_options: Dict[str, Any] = {},
        control: Optional[Dict[str, List[Optional[torch.Tensor]]]] = None,
        **kwargs,
    ) -> torch.Tensor:

        device = x.device
        if self.offload and self.offload_manager is not None:
            self.offload_manager.set_device(device)

        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        if ref_latents is not None:
            h = w = index = 0
            index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
            for ref in ref_latents:
                if index_ref_method:
                    index += 1
                    h_offset = 0
                    w_offset = 0
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2,
                              ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
        txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)
        del ids, txt_ids, img_ids

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        temb = (self.time_text_embed(timestep, hidden_states) if guidance is None
                else self.time_text_embed(timestep, guidance, hidden_states))

        # ---- patches + control ----
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = {}
        for k in ("dit", "qwen", "double_blocks", "blocks"):
            v = patches_replace.get(k, {})
            if isinstance(v, dict):
                blocks_replace.update(v)

        if control is None:
            control = kwargs.get("control", None)
        if control is None:
            control = transformer_options.get("control", None)

        # Optional scale
        try:
            control_scale = float(control.get("weight", control.get("scale", 1.0))) if control is not None else 1.0
        except Exception:
            control_scale = 1.0

        # Pre-get lists (may be None)
        c_input_img = control.get("input", None) if isinstance(control, dict) else None
        c_input_txt = control.get("input_txt", None) if isinstance(control, dict) else None
        c_output_img = control.get("output", None) if isinstance(control, dict) else None
        c_output_txt = control.get("output_txt", None) if isinstance(control, dict) else None

        compute_stream = torch.cuda.current_stream()
        if self.offload and self.offload_manager is not None:
            self.offload_manager.initialize(compute_stream)

        num_layers = len(self.transformer_blocks)

        for i, block in enumerate(self.transformer_blocks):
            with torch.cuda.stream(compute_stream):
                if self.offload and self.offload_manager is not None:
                    block = self.offload_manager.get_block(i)

                # prepare patch args
                args = {
                    "img": hidden_states,
                    "txt": encoder_hidden_states,
                    "vec": temb,
                    "pe": image_rotary_emb,
                    "mask": encoder_hidden_states_mask,
                    "block_index": i,
                    "num_layers": num_layers,
                    "timestep": timestep,
                }

                def _orig(a):
                    t_out, i_out = block(
                        hidden_states=a.get("img", hidden_states),
                        encoder_hidden_states=a.get("txt", encoder_hidden_states),
                        encoder_hidden_states_mask=a.get("mask", encoder_hidden_states_mask),
                        temb=a.get("vec", temb),
                        image_rotary_emb=a.get("pe", image_rotary_emb),
                    )
                    return {"img": i_out, "txt": t_out}

                patch_fn = (
                    blocks_replace.get(("double_block", i), None)
                    or blocks_replace.get(("block", i), None)
                    or blocks_replace.get(i, None)
                )

                if patch_fn is not None:
                    try:
                        out = patch_fn(args, {"original_block": _orig})
                    except TypeError:
                        out = patch_fn(args)
                    if isinstance(out, dict):
                        hidden_states = out.get("img", hidden_states)
                        encoder_hidden_states = out.get("txt", encoder_hidden_states)
                    else:
                        try:
                            enc_out, img_out = out
                            if img_out is not None:
                                hidden_states = img_out
                            if enc_out is not None:
                                encoder_hidden_states = enc_out
                        except Exception:
                            pass
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                    )

                # ---- ControlNet per-layer residuals (keep offload; align device/dtype non_blocking) ----
                if c_input_img is not None and i < len(c_input_img):
                    add = _to_dev_dtype(c_input_img[i], hidden_states)
                    if add is not None:
                        # crop to token overlap
                        if add.dim() == 3 and hidden_states.dim() == 3 and add.shape[1] > hidden_states.shape[1]:
                            add = add[:, :hidden_states.shape[1], :]
                        _apply_residual(hidden_states, add, control_scale)

                if c_input_txt is not None and i < len(c_input_txt):
                    addt = _to_dev_dtype(c_input_txt[i], encoder_hidden_states)
                    if addt is not None:
                        if addt.dim() == 3 and encoder_hidden_states.dim() == 3 and addt.shape[1] > encoder_hidden_states.shape[1]:
                            addt = addt[:, :encoder_hidden_states.shape[1], :]
                        _apply_residual(encoder_hidden_states, addt, control_scale)

                if c_output_img is not None and i < len(c_output_img):
                    addo = _to_dev_dtype(c_output_img[i], hidden_states)
                    if addo is not None:
                        if addo.dim() == 3 and hidden_states.dim() == 3 and addo.shape[1] > hidden_states.shape[1]:
                            addo = addo[:, :hidden_states.shape[1], :]
                        _apply_residual(hidden_states, addo, control_scale)

                if c_output_txt is not None and i < len(c_output_txt):
                    addot = _to_dev_dtype(c_output_txt[i], encoder_hidden_states)
                    if addot is not None:
                        if addot.dim() == 3 and encoder_hidden_states.dim() == 3 and addot.shape[1] > encoder_hidden_states.shape[1]:
                            addot = addot[:, :encoder_hidden_states.shape[1], :]
                        _apply_residual(encoder_hidden_states, addot, control_scale)

            if self.offload and self.offload_manager is not None:
                self.offload_manager.step(compute_stream)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds].view(
            orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2
        )
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, :, : x.shape[-2], : x.shape[-1]]

    def set_offload(self, offload: bool, **kwargs):
        if offload == getattr(self, "offload", False):
            return
        self.offload = offload
        if offload:
            self.offload_manager = CPUOffloadManager(
                self.transformer_blocks,
                use_pin_memory=kwargs.get("use_pin_memory", True),
                on_gpu_modules=[self.img_in, self.txt_in, self.txt_norm, self.time_text_embed, self.norm_out, self.proj_out],
                num_blocks_on_gpu=kwargs.get("num_blocks_on_gpu", 1),
            )
        else:
            self.offload_manager = None
            gc.collect()
            torch.cuda.empty_cache()
