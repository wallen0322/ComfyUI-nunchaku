"""
This module wraps the ComfyUI model patcher for Nunchaku models to load and unload the model correctly.
"""

import comfy.model_patcher


class NunchakuModelPatcher(comfy.model_patcher.ModelPatcher):
    """
    This class extends the ComfyUI ModelPatcher to provide custom logic for loading and unloading the model correctly.
    """

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        """
        Load the diffusion model onto the specified device.

        Parameters
        ----------
        device_to : torch.device or str, optional
            The device to which the diffusion model should be moved.
        lowvram_model_memory : int, optional
            Not used in this implementation.
        force_patch_weights : bool, optional
            Not used in this implementation.
        full_load : bool, optional
            Not used in this implementation.
        """
        with self.use_ejected():
            self.model.diffusion_model.to_safely(device_to)

    def detach(self, unpatch_all: bool = True):
        """
        Detach the model and move it to the offload device.

        Parameters
        ----------
        unpatch_all : bool, optional
            If True, unpatch all model components (default is True).
        """
        self.eject_model()
        self.model.diffusion_model.to_safely(self.offload_device)

# ---------- Minimal LoRA helpers (quant-safe, forward-add) ----------
# These utilities are intentionally small and self-contained.
# They let us attach LoRA in a quantization-friendly way without touching kernels.

from typing import Any, List, Optional, Tuple
import types
import torch
import torch.nn as nn

def _get_submodule(root: nn.Module, dotted: str) -> Optional[nn.Module]:
    """
    Resolve a dotted path (e.g., 'blocks.0.attn.to_qkv') to a submodule.
    Returns None if not found.
    """
    cur = root
    if not dotted:
        return cur
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur

def _ensure_lora_wrap_linear(mod: nn.Module):
    """
    Monkey-patch mod.forward once to support LoRA forward-add hooks.
    LoRA entries are stored on mod._nch_lora_entries as a list of dicts.
    """
    if getattr(mod, "_nch_lora_patched", False):
        return

    orig_forward = mod.forward

    def _forward_with_lora(x: torch.Tensor, *args, **kwargs):
        y = orig_forward(x, *args, **kwargs)
        entries = getattr(mod, "_nch_lora_entries", None)
        if not entries:
            return y

        x_in = x

        for ent in entries:
            A = ent["A"]
            B = ent["B"]
            scale = ent["scale"]
            slice_kind = ent["slice_kind"]
            is_qkv = ent["is_qkv"]

            A_ = A.to(dtype=x_in.dtype, device=x_in.device)
            B_ = B.to(dtype=x_in.dtype, device=x_in.device)

            # z = B(A(x))   (x: [..., in], A: [r, in], B: [out, r])
            z = torch.nn.functional.linear(x_in, A_.transpose(0, 1))   # [..., r]
            z = torch.nn.functional.linear(z,   B_.transpose(0, 1))     # [..., out]

            if is_qkv and slice_kind is not None:
                out_dim = y.shape[-1]
                if out_dim % 3 == 0:
                    chunk = out_dim // 3
                    idx = {"q": 0, "k": 1, "v": 2}[slice_kind]
                    s = idx * chunk
                    e = s + chunk
                    y[..., s:e] = y[..., s:e] + scale * z[..., :chunk]
                else:
                    y = y + scale * z
            else:
                y = y + scale * z

        return y

    mod._nch_lora_entries = []
    mod._nch_lora_patched = True
    mod.forward = types.MethodType(_forward_with_lora, mod)

def _register_lora_on_linear(
    mod: nn.Module,
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float,
    slice_kind: Optional[str] = None,
    is_qkv: bool = False,
):
    """
    Register one LoRA entry onto a Linear-like module.
    """
    _ensure_lora_wrap_linear(mod)
    if not hasattr(mod, "_nch_lora_buffers"):
        mod._nch_lora_buffers = []
    A_b = A.detach().clone()
    B_b = B.detach().clone()
    mod._nch_lora_buffers.append(A_b)
    mod._nch_lora_buffers.append(B_b)

    entry = {
        "A": A_b,
        "B": B_b,
        "scale": float(scale),
        "slice_kind": slice_kind,     # None | 'q' | 'k' | 'v'
        "is_qkv": bool(is_qkv),
    }
    mod._nch_lora_entries.append(entry)

def attach_lora_entries(
    model: nn.Module,
    entries: List[Tuple[str, str, torch.Tensor, torch.Tensor, float, float, Optional[str]]],
    multiplier: float = 1.0,
):
    """
    Attach LoRA to a model (quantization-safe, forward-add). Expected entry tuple:

        (kind, weight_path, A, B, alpha, strength, slice_or_None)

    kind ∈ {'linear','qkv','split'}; weight_path ends with '.weight'.
    slice_or_None ∈ {'q','k','v',None}; for kind 'qkv' we'll add to the 1/3 slice.
    """
    for kind, weight_path, A, B, alpha, strength, slice_kind in entries:
        if not isinstance(weight_path, str) or not weight_path.endswith(".weight"):
            continue
        module_path = weight_path[: -len(".weight")]
        mod = _get_submodule(model, module_path)
        if mod is None or not hasattr(mod, "forward"):
            continue
        rank = float(A.shape[0]) if A.dim() == 2 else 1.0
        scale = float(strength) * float(alpha) / max(rank, 1.0)
        is_qkv = (kind == "qkv")
        _register_lora_on_linear(
            mod,
            A=A,
            B=B,
            scale=scale * float(multiplier),
            slice_kind=(slice_kind if slice_kind in ("q", "k", "v") else None),
            is_qkv=is_qkv,
        )
