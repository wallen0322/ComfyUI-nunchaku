
# wrappers/qwenimage.py  (fix16.5: functional-LoRA QKV weight-patch; others still hook; auto-merge)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Iterable, Tuple, List, Optional

Record = Tuple[str, str, torch.Tensor, torch.Tensor, float, float, Any]

_QKV_FUSED_SUFFIXES = (".to_qkv", ".add_qkv_proj", ".W_pack", ".in_proj", ".c_attn", ".qkv_proj", ".proj_qkv")
_FUSED_WEIGHT_CAND_NAMES = ("weight", "in_proj_weight", "qkv_weight", "Wqkv", "proj_weight")

def _unwrap_module(obj) -> Optional[nn.Module]:
    if isinstance(obj, nn.Module):
        return obj
    for name in ("model", "text_encoder", "clip", "transformer", "encoder", "net", "backbone"):
        inner = getattr(obj, name, None)
        if isinstance(inner, nn.Module):
            return inner
    for name in ("model", "text_encoder", "clip", "transformer", "encoder", "net", "backbone"):
        inner = getattr(obj, name, None)
        if inner is not None:
            m = _unwrap_module(inner)
            if isinstance(m, nn.Module):
                return m
    return None

def _module_suffix_index(root: nn.Module) -> List[str]:
    names = [n for n, _ in root.named_modules()]
    names.sort(key=len, reverse=True)
    return names

def _resolve_module_name(root: nn.Module, short_key: str, idx: List[str]):
    short_module = short_key[:-7] if short_key.endswith(".weight") else short_key
    for full in idx:
        if full.endswith(short_module):
            return full
    return None

def _resolve_any(root: nn.Module, candidates: List[str], idx: List[str]) -> Optional[str]:
    for c in candidates:
        m = _resolve_module_name(root, c, idx)
        if m is not None:
            return m
    return None

def _parent_from_qkv_target(fused_key: str) -> str:
    key = fused_key[:-7] if fused_key.endswith(".weight") else fused_key
    for suf in _QKV_FUSED_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)]
    if "." in key:
        return key.rsplit(".", 1)[0]
    return key

def _clear_old_hooks(root: nn.Module):
    hooks = getattr(root, "_qwen_lora_hooks", None)
    if hooks:
        for h in hooks:
            try: h.remove()
            except Exception: pass
    root._qwen_lora_hooks = []

def _stack_triples(triples: List[Tuple[torch.Tensor, torch.Tensor, float]]):
    if not triples: return (None, None)
    A_list, B_list = [], []
    for (A,B,s) in triples:
        A_list.append(A.detach().contiguous())
        B_list.append(B.detach().contiguous() * float(s))
    Acat = torch.cat(A_list, dim=0)  # [R_sum, in]
    Bcat = torch.cat(B_list, dim=1)  # [out, R_sum]
    return (Acat, Bcat)

class _PerDeviceCache:
    def __init__(self):
        self.store = {}  # (id, device, dtype) -> tensor
    def get(self, key):
        return self.store.get(key, None)
    def set(self, key, val):
        self.store[key] = val

def _to_cached(t: Optional[torch.Tensor], device, dtype, cache: _PerDeviceCache):
    if t is None: return None
    k = (id(t), device, dtype)
    got = cache.get(k)
    if got is not None: return got
    moved = t.to(device=device, dtype=dtype, non_blocking=True)
    cache.set(k, moved)
    return moved

def _maybe_detect_qkv_order(mod: nn.Module, out_dim: int) -> Optional[Tuple[int,int,int]]:
    try:
        w = None
        for nm in _FUSED_WEIGHT_CAND_NAMES:
            if hasattr(mod, nm):
                tw = getattr(mod, nm)
                if isinstance(tw, torch.Tensor) and tw.ndim == 2 and tw.shape[0] == 3*out_dim:
                    w = tw; break
        if w is None: return None
        parent = mod
        cand = {}
        for n in ("to_q", "q_proj", "query", "wq"):
            tw = getattr(parent, n, None)
            if isinstance(tw, nn.Linear): cand["q"] = tw.weight.data
        for n in ("to_k", "k_proj", "key", "wk"):
            tw = getattr(parent, n, None)
            if isinstance(tw, nn.Linear): cand.setdefault("k", tw.weight.data)
        for n in ("to_v", "v_proj", "value", "wv"):
            tw = getattr(parent, n, None)
            if isinstance(tw, nn.Linear): cand.setdefault("v", tw.weight.data)
        if len(cand) < 1: return None
        chunks = w.chunk(3, dim=0)
        def _sim(a, b):
            a2 = a.reshape(a.shape[0], -1); b2 = b.reshape(b.shape[0], -1)
            num = torch.sum(a2*b2)
            den = (torch.linalg.norm(a2)+1e-6)*(torch.linalg.norm(b2)+1e-6)
            return float(num/den)
        picks = []
        for ch in chunks:
            sq = _sim(ch, cand.get("q", ch*0))
            sk = _sim(ch, cand.get("k", ch*0))
            sv = _sim(ch, cand.get("v", ch*0))
            picks.append(int(torch.tensor([sq, sk, sv]).argmax().item()))
        perm = tuple(picks)
        if set(perm) == {0,1,2}: return perm
        return None
    except Exception:
        return None

def _fused_weight_attr(mod: nn.Module, out_dim: int) -> Optional[str]:
    for nm in _FUSED_WEIGHT_CAND_NAMES:
        if hasattr(mod, nm):
            tw = getattr(mod, nm)
            if isinstance(tw, torch.Tensor) and tw.ndim == 2 and tw.shape[0] == 3*out_dim:
                return nm
    return None

def _attach_lowrank_hooks(root: nn.Module, factors: Dict[str, Dict[str, Any]], functional=False):
    name_to_module = {name: module for name, module in root.named_modules()}
    hooks: List[torch.utils.hooks.RemovableHandle] = []
    attached = 0
    samples: List[str] = []
    cache = _PerDeviceCache()

    for mod_name, spec in factors.items():
        mod = name_to_module.get(mod_name, None)
        if mod is None:
            continue

        # --- QKV functional → weight patch (preferred) ---
        if functional and "qkv" in spec and spec["qkv"]:
            info = spec["qkv"]; out_dim = int(info["out"])
            A_q,B_q = _stack_triples(info.get("q", []))
            A_k,B_k = _stack_triples(info.get("k", []))
            A_v,B_v = _stack_triples(info.get("v", []))
            order = _maybe_detect_qkv_order(mod, out_dim) or (0,1,2)
            attr = _fused_weight_attr(mod, out_dim)
            if attr is not None:
                base_w = getattr(mod, attr)
                device, dtype = base_w.device, base_w.dtype
                # build delta on device
                def mat(Ac,Bc):
                    if Ac is None or Bc is None: return None
                    A = _to_cached(Ac, device, dtype, cache)
                    B = _to_cached(Bc, device, dtype, cache)
                    return (B @ A).to(dtype=dtype)
                dq = mat(A_q,B_q); dk = mat(A_k,B_k); dv = mat(A_v,B_v)
                delta = torch.zeros_like(base_w)
                idx = {0: order.index(0), 1: order.index(1), 2: order.index(2)}
                if dq is not None: delta[idx[0]*out_dim:(idx[0]+1)*out_dim, :] += dq
                if dk is not None: delta[idx[1]*out_dim:(idx[1]+1)*out_dim, :] += dk
                if dv is not None: delta[idx[2]*out_dim:(idx[2]+1)*out_dim, :] += dv
                with torch.no_grad():
                    base_w.add_(delta)
                attached += 1
                if len(samples) < 8: samples.append(mod_name + "#patch")
                continue  # skip adding forward hook for this qkv

        # --- Linear or non-functional path → classic forward add ---
        if "linear" in spec and spec["linear"]:
            Acat, Bcat = _stack_triples([(A,B,s) for (A,B,s) in spec["linear"]])
            def make_hook_linear(Acat, Bcat):
                def hook(mod, inputs, output):
                    try:
                        if Acat is None or Bcat is None: return output
                        x = inputs[0]; x2 = x.reshape(-1, x.shape[-1]).contiguous()
                        A = _to_cached(Acat, x2.device, x2.dtype, cache)
                        B = _to_cached(Bcat, x2.device, x2.dtype, cache)
                        add = F.linear(F.linear(x2, A), B)
                        add = add.view(*x.shape[:-1], add.shape[-1])
                        return output + add
                    except Exception:
                        return output
                return mod.register_forward_hook(hook)
            h = make_hook_linear(Acat, Bcat)
            hooks.append(h); attached += 1
            if len(samples) < 8: samples.append(mod_name)

        if (not functional) and "qkv" in spec and spec["qkv"]:
            info = spec["qkv"]; out_dim = int(info["out"])
            A_q,B_q = _stack_triples(info.get("q", []))
            A_k,B_k = _stack_triples(info.get("k", []))
            A_v,B_v = _stack_triples(info.get("v", []))
            order = None
            def make_hook_qkv(A_q, B_q, A_k, B_k, A_v, B_v, out_dim):
                def hook(mod, inputs, output):
                    nonlocal order
                    try:
                        x = inputs[0]; x2 = x.reshape(-1, x.shape[-1]).contiguous()
                        if order is None:
                            order = _maybe_detect_qkv_order(mod, out_dim) or (0,1,2)
                        def add_branch(Ac, Bc):
                            if Ac is None or Bc is None: return None
                            A = _to_cached(Ac, x2.device, x2.dtype, cache)
                            B = _to_cached(Bc, x2.device, x2.dtype, cache)
                            return F.linear(F.linear(x2, A), B)
                        if isinstance(output, torch.Tensor) and output.shape[-1] == (out_dim * 3):
                            N = x2.shape[0]
                            add = output.new_zeros((N, out_dim * 3))
                            aq = add_branch(A_q,B_q); ak = add_branch(A_k,B_k); av = add_branch(A_v,B_v)
                            if aq is not None: add[:, order.index(0)*out_dim:(order.index(0)+1)*out_dim] += aq
                            if ak is not None: add[:, order.index(1)*out_dim:(order.index(1)+1)*out_dim] += ak
                            if av is not None: add[:, order.index(2)*out_dim:(order.index(2)+1)*out_dim] += av
                            add = add.view(*x.shape[:-1], add.shape[-1])
                            return output + add
                        if isinstance(output, (tuple, list)) and len(output) == 3:
                            q,k,v = output
                            aq = add_branch(A_q,B_q); ak = add_branch(A_k,B_k); av = add_branch(A_v,B_v)
                            if aq is not None: q = q + aq.view(*q.shape)
                            if ak is not None: k = k + ak.view(*k.shape)
                            if av is not None: v = v + av.view(*v.shape)
                            return (q,k,v)
                        return output
                    except Exception:
                        return output
                return mod.register_forward_hook(hook)
            h = make_hook_qkv(A_q,B_q,A_k,B_k,A_v,B_v,out_dim)
            hooks.append(h); attached += 1
            if len(samples) < 8: samples.append(mod_name)

    root._qwen_lora_hooks = hooks
    print(f"[Qwen-LoRA][hooks] attached={attached}")
    if samples:
        print(f"[Qwen-LoRA][hooks] sample attached: {samples}")

def set_qwen_lora_options(model_or_wrapper, **cfg):
    root = _unwrap_module(model_or_wrapper) or model_or_wrapper
    setattr(root, "_qwen_lora_cfg", dict(cfg))

def update_model_with_qwen_lora(model_or_wrapper, records: Iterable[Record]):
    root = _unwrap_module(model_or_wrapper)
    if root is None:
        print("[Qwen-LoRA] WARN: cannot unwrap to nn.Module; skip")
        return

    cfg = getattr(root, "_qwen_lora_cfg", {}) or getattr(model_or_wrapper, "_qwen_lora_cfg", {}) or {}

    # auto-merge: first reset, then merge
    if "reset" in cfg:
        reset = bool(cfg.get("reset"))
    else:
        prev_exist = hasattr(root, "_qwen_lora_records") and bool(getattr(root, "_qwen_lora_records"))
        reset = not prev_exist
    mode = "reset" if reset else "merge"
    print(f"[Qwen-LoRA] apply mode: {mode}")
    boost_qkv = bool(cfg.get("boost_qkv", False))
    functional = bool(cfg.get("functional", False))

    prev: List[Record] = getattr(root, "_qwen_lora_records", [])
    if reset:
        records_to_use: List[Record] = list(records or [])
    else:
        cache = {}
        for r in (prev or []):
            k = (r[0], r[1], r[6]); cache[k] = r
        for r in (records or []):
            k = (r[0], r[1], r[6]); cache[k] = r
        records_to_use = list(cache.values())

    idx = _module_suffix_index(root)
    factors: Dict[str, Dict[str, Any]] = {}
    miss = {"linear": 0, "q": 0, "k": 0, "v": 0, "fused": 0}
    reroute = {"q": 0, "k": 0, "v": 0}

    def add_linear(short_key, A, B, s):
        mod_name = _resolve_module_name(root, short_key, idx)
        if mod_name is not None:
            spec = factors.setdefault(mod_name, {})
            spec.setdefault("linear", []).append((A, B, s))
            return True
        miss["linear"] += 1
        return False

    def add_qkv_or_fallback(short_key, A, B, s, which):
        parent = _parent_from_qkv_target(short_key)

        def add_to_split():
            cand_map = {
                "q": [parent + ".q_proj", parent + ".query", parent + ".q", parent + ".wq", parent + ".to_q"],
                "k": [parent + ".k_proj", parent + ".key",   parent + ".k", parent + ".wk", parent + ".to_k"],
                "v": [parent + ".v_proj", parent + ".value", parent + ".v", parent + ".wv", parent + ".to_v"],
            }
            target = _resolve_any(root, cand_map.get(which, []), idx)
            if target is not None:
                spec = factors.setdefault(target, {})
                spec.setdefault("linear", []).append((A, B, s))
                return True
            return False

        def add_to_fused():
            mod_name = _resolve_module_name(root, short_key, idx)
            if mod_name is not None:
                spec = factors.setdefault(mod_name, {})
                info = spec.setdefault("qkv", {"q": [], "k": [], "v": [], "out": int(B.shape[0]), "in": int(A.shape[1])})
                info[which].append((A, B, s))
                return True
            fused_cands = [parent + suf for suf in (".W_pack", ".in_proj", ".c_attn", ".to_qkv", ".add_qkv_proj", ".qkv_proj", ".proj_qkv")]
            target = _resolve_any(root, fused_cands, idx)
            if target is not None:
                spec = factors.setdefault(target, {})
                info = spec.setdefault("qkv", {"q": [], "k": [], "v": [], "out": int(B.shape[0]), "in": int(A.shape[1])})
                info[which].append((A, B, s))
                return True
            return False

        if add_to_split():
            return True
        if add_to_fused():
            reroute[which] += 1
            return True
        miss[which] += 1
        return False

    def _scale_local(alpha, A, strength):
        r = max(1.0, float(A.shape[0]))
        gain = float(alpha) / r * float(strength)
        if boost_qkv:
            gain *= 1.75
        return max(-16.0, min(16.0, gain))

    for kind, path, A, B, alpha, strength, which in (records_to_use or []):
        s = _scale_local(alpha, A, strength)
        if kind == "qkv":
            add_qkv_or_fallback(path, A, B, s, which)
        else:
            add_linear(path, A, B, s)

    _clear_old_hooks(root)
    _attach_lowrank_hooks(root, factors, functional=functional)
    setattr(root, "_qwen_lora_records", records_to_use)

    if any(v > 0 for v in reroute.values()):
        print(f"[Qwen-LoRA][diagnostics] split->fused rerouted: {reroute}")
    if sum(miss.values()) > 0:
        print(f"[Qwen-LoRA][diagnostics] unmapped parts: {miss} (non-fatal)")

    try: delattr(root, "_qwen_lora_cfg")
    except Exception: pass
