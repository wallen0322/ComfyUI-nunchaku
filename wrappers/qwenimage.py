
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Iterable, Tuple, List, Optional, Set

Record = Tuple[str, str, torch.Tensor, torch.Tensor, float, float, Any]  # (kind, path, A, B, alpha, strength, which)

def _scale(alpha: float, A: torch.Tensor, strength: float) -> float:
    r = max(1.0, float(A.shape[0]))
    return float(alpha) / r * float(strength)

def _unwrap_module(obj) -> Optional[nn.Module]:
    """Try hard to find a real nn.Module inside wrappers (MODEL/CLIP)."""
    if isinstance(obj, nn.Module):
        return obj
    # common attributes
    for name in ("model", "text_encoder", "clip", "transformer", "encoder", "net", "backbone"):
        inner = getattr(obj, name, None)
        if isinstance(inner, nn.Module):
            return inner
    # one more level
    for name in ("model", "text_encoder", "clip", "transformer", "encoder", "net", "backbone"):
        inner = getattr(obj, name, None)
        if inner is not None:
            m = _unwrap_module(inner)
            if isinstance(m, nn.Module):
                return m
    # generic deep search (shallow)
    seen: Set[int] = set()
    def dfs(o, depth=0):
        if depth > 2:  # avoid runaway
            return None
        oid = id(o)
        if oid in seen: return None
        seen.add(oid)
        if isinstance(o, nn.Module):
            return o
        for attr in dir(o):
            if attr.startswith("_"): continue
            try:
                v = getattr(o, attr)
            except Exception:
                continue
            if isinstance(v, nn.Module):
                return v
            if hasattr(v, "__dict__"):
                m = dfs(v, depth+1)
                if isinstance(m, nn.Module):
                    return m
        return None
    return dfs(obj, 0)

def _module_suffix_index(root: nn.Module) -> List[str]:
    names = [n for n, _ in root.named_modules()]
    names.sort(key=len, reverse=True)
    return names

def _resolve_module_name(root: nn.Module, short_key: str, idx: List[str]):
    if short_key.endswith(".weight"):
        short_module = short_key[:-7]
    else:
        short_module = short_key
    for full in idx:
        if full.endswith(short_module):
            return full
    return None

def _clear_old_hooks(root: nn.Module):
    hooks = getattr(root, "_qwen_lora_hooks", None)
    if hooks:
        for h in hooks:
            try: h.remove()
            except Exception: pass
    root._qwen_lora_hooks = []

def _move_once(buf: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if buf.device == ref.device and buf.dtype == ref.dtype:
        return buf
    return buf.to(device=ref.device, dtype=ref.dtype, non_blocking=True)

def _attach_lowrank_hooks(root: nn.Module, factors: Dict[str, Dict[str, Any]]):
    name_to_module = {name: module for name, module in root.named_modules()}

    hooks: List[torch.utils.hooks.RemovableHandle] = []
    attached = 0
    samples = []

    for mod_name, spec in factors.items():
        mod = name_to_module.get(mod_name, None)
        if mod is None:
            continue

        if "linear" in spec and spec["linear"]:
            triples = [(A.contiguous(), B.contiguous(), float(s)) for (A,B,s) in spec["linear"]]

            def make_hook_linear(tris):
                def hook(module, inputs, output):
                    try:
                        x = inputs[0]
                        x2 = x.view(-1, x.shape[-1])
                        add = None
                        for (A, B, s) in tris:
                            Ax  = F.linear(x2, _move_once(A, x2))
                            BAx = F.linear(Ax, _move_once(B, x2))
                            add = (BAx.mul_(s) if add is None else add + BAx.mul_(s))
                        if add is None: return output
                        add = add.view(*x.shape[:-1], add.shape[-1])
                        return output + add
                    except Exception:
                        return output
                return mod.register_forward_hook(hook)

            h = make_hook_linear(triples)
            hooks.append(h); attached += 1
            if len(samples) < 8: samples.append(mod_name + ".weight")

        if "qkv" in spec and spec["qkv"]:
            info = spec["qkv"]
            out_dim = int(info["out"])
            q_tris = [(A.contiguous(), B.contiguous(), float(s)) for (A,B,s) in info.get("q", [])]
            k_tris = [(A.contiguous(), B.contiguous(), float(s)) for (A,B,s) in info.get("k", [])]
            v_tris = [(A.contiguous(), B.contiguous(), float(s)) for (A,B,s) in info.get("v", [])]

            def make_hook_qkv(qt, kt, vt, out_dim):
                def hook(module, inputs, output):
                    try:
                        x = inputs[0]
                        x2 = x.view(-1, x.shape[-1])
                        def acc(tris):
                            add = None
                            for (A, B, s) in tris:
                                Ax  = F.linear(x2, _move_once(A, x2))
                                BAx = F.linear(Ax, _move_once(B, x2))
                                add = (BAx.mul_(s) if add is None else add + BAx.mul_(s))
                            return add
                        add_q = acc(qt)
                        add_k = acc(kt)
                        add_v = acc(vt)
                        if add_q is None and add_k is None and add_v is None:
                            return output
                        N = x2.shape[0]
                        add = output.new_zeros((N, out_dim * 3))
                        if add_q is not None: add[:, 0:out_dim] += add_q
                        if add_k is not None: add[:, out_dim:2*out_dim] += add_k
                        if add_v is not None: add[:, 2*out_dim:3*out_dim] += add_v
                        add = add.view(*x.shape[:-1], add.shape[-1])
                        return output + add
                    except Exception:
                        return output
                return mod.register_forward_hook(hook)  # <-- fix: use 'mod', not 'module'

            h = make_hook_qkv(q_tris, k_tris, v_tris, out_dim)
            hooks.append(h); attached += 1
            if len(samples) < 8: samples.append(mod_name + ".weight")

    root._qwen_lora_hooks = hooks
    print(f"[Qwen-LoRA][hooks] attached={attached}")
    if samples:
        print(f"[Qwen-LoRA][hooks] sample attached: {samples}")

def update_model_with_qwen_lora(model_or_wrapper, records: Iterable[Record]):
    root = _unwrap_module(model_or_wrapper)
    if root is None:
        print("[Qwen-LoRA] WARN: cannot unwrap to nn.Module; skip")
        return

    idx = _module_suffix_index(root)
    factors: Dict[str, Dict[str, Any]] = {}

    def add_linear(short_key, A, B, s):
        mod_name = _resolve_module_name(root, short_key, idx)
        if mod_name is None: return
        spec = factors.setdefault(mod_name, {})
        spec.setdefault("linear", []).append((A, B, s))

    def add_qkv(short_key, A, B, s, which):
        mod_name = _resolve_module_name(root, short_key, idx)
        if mod_name is None: return
        spec = factors.setdefault(mod_name, {})
        info = spec.setdefault("qkv", {"q": [], "k": [], "v": [], "out": int(B.shape[0]), "in": int(A.shape[1])})
        info[which].append((A, B, s))

    for kind, path, A, B, alpha, strength, which in (records or []):
        s = _scale(alpha, A, strength)
        if kind == "qkv":
            add_qkv(path, A, B, s, which)
        else:
            add_linear(path, A, B, s)

    _clear_old_hooks(root)
    _attach_lowrank_hooks(root, factors)
