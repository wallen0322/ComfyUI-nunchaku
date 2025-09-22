# nodes/lora/qwenimage.py (fix17.1: add quant-safe fallback using model_patcher.attach_lora_entries)
import os, logging, unicodedata, re
import folder_paths
from typing import Dict, Any, List, Tuple, Optional
from safetensors.torch import load_file

# 优先使用包装器实现（若量化包装器已支持），否则回退到本地 forward-hook
_use_wrapper = True
try:
    from ...wrappers.qwenimage import update_model_with_qwen_lora, set_qwen_lora_options
except Exception:
    update_model_with_qwen_lora = None
    set_qwen_lora_options = None
    _use_wrapper = False

# 本地兜底：量化安全的前向叠加（不接触权重）
try:
    from ...model_patcher import attach_lora_entries as _attach_lora_entries
except Exception:
    _attach_lora_entries = None

import torch

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PUNC_MAP = {"。": ".", "．": ".", "・": ".", "·": ".", "｡": ".", "﹒": "."}
WORD_MAP = {"阿尔法": "alpha", "α": "alpha"}
SUFFIX_RE = re.compile(r'\.(?:lora_A|lora_B|lora_down|lora_up)(?:\.[A-Za-z0-9_]+)?\.weight$')

def _normalize_key(k: str) -> str:
    k = unicodedata.normalize("NFKC", k)
    for a,b in PUNC_MAP.items(): k = k.replace(a,b)
    for a,b in WORD_MAP.items(): k = k.replace(a,b)
    return k

def _load_lora(path: str) -> Dict[str, Any]:
    sd = load_file(path)
    return {_normalize_key(k): v for k, v in sd.items()}

def _collect_bases(sd: Dict[str, Any]):
    bases = set()
    for k in sd.keys():
        if SUFFIX_RE.search(k):
            parts = k.split(".")
            if len(parts)>=3 and parts[-2]=="default":
                base = ".".join(parts[:-3])
            else:
                base = ".".join(parts[:-2])
            bases.add(base)
    return bases

def _fetch(sd: Dict[str, Any], *cands: str):
    for c in cands:
        if c in sd: return sd[c]
    return None

def _to_qwen_records(sd: Dict[str, Any], strength: float):
    recs = []
    bases = _collect_bases(sd)
    if not bases:
        keys = list(sd.keys())
        logger.warning("[Qwen-LoRA] 0 bases parsed. sample keys=%s", keys[:40])
        return recs

    qkv_last = {"to_q","to_k","to_v","add_q_proj","add_k_proj","add_v_proj","q_proj","k_proj","v_proj"}
    for base in sorted(bases):
        A = _fetch(sd,
                   f"{base}.lora_A.weight", f"{base}.lora_A.default.weight",
                   f"{base}.lora_down.weight", f"{base}.lora_down.default.weight")
        B = _fetch(sd,
                   f"{base}.lora_B.weight", f"{base}.lora_B.default.weight",
                   f"{base}.lora_up.weight", f"{base}.lora_up.default.weight")
        if A is None or B is None: continue
        alpha_t = _fetch(sd, f"{base}.alpha", f"{base}.alpha.default")
        try: alpha = float(alpha_t.item()) if hasattr(alpha_t, "item") else float(A.shape[0])
        except Exception: alpha = float(A.shape[0])

        last = base.split(".")[-1]
        parent = ".".join(base.split(".")[:-1])

        if last in qkv_last:
            # split 精确路径
            recs.append(("split", base + ".weight", A, B, alpha, float(strength),
                         "q" if last in ("to_q","add_q_proj","q_proj") else ("k" if last in ("to_k","add_k_proj","k_proj") else "v")))
            # fused 兜底
            fused = parent + (".add_qkv_proj.weight" if last.startswith("add_") else ".to_qkv.weight")
            recs.append(("qkv", fused, A, B, alpha, float(strength),
                         "q" if last in ("to_q","add_q_proj","q_proj") else ("k" if last in ("to_k","add_k_proj","k_proj") else "v")))
        else:
            recs.append(("linear", base + ".weight", A, B, alpha, float(strength), None))

    logger.info(f"[Qwen-LoRA] parsed bases={len(bases)}, qkv_pairs={sum(1 for r in recs if r[0]=='qkv')}, linear_pairs={sum(1 for r in recs if r[0]=='linear')}")
    return recs

def _discover_clip_from_model(model_obj):
    for name in ("text_encoder","clip","te","t5","qwen_text","qwen_te","transformer","encoder","cond_stage_model","context_encoder","text_model"):
        if hasattr(model_obj, name): return getattr(model_obj, name)
    inner = getattr(model_obj, "model", None)
    if inner is not None: return _discover_clip_from_model(inner)
    return None

class NunchakuQwenImageLoRALoader:
    TITLE = "Nunchaku Qwen-Image Auto LoRA Loader"
    CATEGORY = "Nunchaku"
    FUNCTION = "load_lora"
    DESCRIPTION = "Qwen-Image LoRA loader (fix17.1): 优先走 wrapper；若量化包装器不支持，则回退到本地前向叠加（含 qkv 1/3 切片），兼容“双截棍”量化。"
    RETURN_TYPES = ("MODEL","CLIP")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {}),
                "model_strength": ("FLOAT", {"default": 1.0, "min": -8.0, "max": 8.0, "step": 0.01}),
                "text_strength": ("FLOAT", {"default": 1.0, "min": -8.0, "max": 8.0, "step": 0.01}),
            },
            "optional": {"model": ("MODEL", {}), "clip": ("CLIP", {})}
        }

    def _apply_with_wrapper(self, target, recs, role, functional=False, boost_qkv=False) -> bool:
        """
        Try wrapper path first. Returns True if applied.
        """
        if target is None or not recs or not _use_wrapper or update_model_with_qwen_lora is None:
            return False
        try:
            if set_qwen_lora_options is not None:
                set_qwen_lora_options(target, functional=functional, boost_qkv=bool(boost_qkv))
        except Exception:
            pass
        try:
            w = getattr(target, "model", None) or getattr(target, "text_encoder", None) or target
            update_model_with_qwen_lora(w, recs)
            logger.info(f"[Qwen-LoRA] applied via WRAPPER to {type(w).__name__} ({role})")
            return True
        except Exception as e:
            logger.warning(f"[Qwen-LoRA] wrapper path failed on {role}: {e}")
            return False

    def _apply_with_fallback(self, target, recs, role) -> bool:
        """
        Fallback: quant-safe forward-add hooks (does not touch weights).
        """
        if target is None or not recs or _attach_lora_entries is None:
            return False
        try:
            w = getattr(target, "model", None) or getattr(target, "text_encoder", None) or target
            _attach_lora_entries(w, recs, multiplier=1.0)
            logger.info(f"[Qwen-LoRA] applied via FALLBACK-HOOK to {type(w).__name__} ({role})")
            return True
        except Exception as e:
            logger.warning(f"[Qwen-LoRA] fallback path failed on {role}: {e}")
            return False

    def load_lora(self, lora_name: str, model_strength: float, text_strength: float, model=None, clip=None):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = _load_lora(lora_path)

        recs_model = _to_qwen_records(sd, model_strength)
        recs_text  = _to_qwen_records(sd, text_strength)

        lname = lora_name.lower()
        is_func = any(k in lname for k in ("light", "lcm", "consist"))

        if is_func:
            logger.info("[Qwen-LoRA] functional/Lightning-ish LoRA detected -> wrapper preferred; fallback hook enabled.")
        else:
            logger.info("[Qwen-LoRA] normal/style LoRA -> wrapper preferred; fallback hook enabled.")

        # MODEL
        # MODEL
        if model is not None:
            logger.info(f"[Qwen-LoRA] MODEL appended {len(recs_model)} records")
            w = getattr(model, "model", None) or getattr(model, "text_encoder", None) or model
            force_hook = _is_quantized_model(w) or _is_chained_loading(w)
            logger.info(f"[Qwen-LoRA] apply mode: {'hook' if force_hook else 'auto'}")
            if not force_hook:
                ok = self._apply_with_wrapper(model, recs_model, role="MODEL", functional=is_func, boost_qkv=is_func)
            else:
                ok = False
            if not ok:
                self._apply_with_fallback(model, recs_model, role="MODEL")
    

            # auto discover TE
            if clip is None:
                auto_clip = _discover_clip_from_model(model)
                if auto_clip is not None:
                    logger.info("[Qwen-LoRA] auto-discovered TE from MODEL")
                    logger.info(f"[Qwen-LoRA] AUTO-CLIP appended {len(recs_text)} records")
                    ok = self._apply_with_wrapper(auto_clip, recs_text, role="AUTO-CLIP", functional=False, boost_qkv=False)
                    if not ok:
                        self._apply_with_fallback(auto_clip, recs_text, role="AUTO-CLIP")

        # explicit CLIP
        if clip is not None:
            logger.info(f"[Qwen-LoRA] CLIP appended {len(recs_text)} records")
            ok = self._apply_with_wrapper(clip, recs_text, role="CLIP", functional=False, boost_qkv=False)
            if not ok:
                self._apply_with_fallback(clip, recs_text, role="CLIP")

        return (model, clip)

# Node registration dictionary for ComfyUI


def _is_quantized_model(m) -> bool:
    # Heuristics: Nunchaku/SVDQuant wrappers usually have these markers
    try:
        t = type(m).__name__
    except Exception:
        t = ""
    names = ("svd", "quant", "SVD", "Quant", "wrapper", "nunchaku")
    return any(s in t for s in names) or hasattr(m, "quant_config") or hasattr(m, "svd_config")

def _is_chained_loading(obj) -> bool:
    # Count how many LoRAs have been attached in this process to the same object
    cnt = getattr(obj, "_nch_lora_load_count", 0)
    setattr(obj, "_nch_lora_load_count", cnt + 1)
    return cnt >= 1  # starting from the second load we consider it "chained"

# ------------------------ Qwen-Image LoRA Stack (wrapper-batched, uniform) ------------------------

class NunchakuQwenImageLoRAStack:
    TITLE = "Nunchaku Qwen-Image LoRA Stack"
    CATEGORY = "Nunchaku"
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL", "CLIP")

    @classmethod
    def INPUT_TYPES(cls):
        opts = ["None"] + folder_paths.get_filename_list("loras")
        required = {"model": ("MODEL", {})}
        optional = {"clip": ("CLIP", {})}
        for i in range(1, 11):
            optional[f"lora_name_{i}"] = (opts, {"default": "None"})
            optional[f"lora_strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -8.0, "max": 8.0, "step": 0.01})
        return {"required": required, "optional": optional}

    def apply(self, model, clip=None, **kwargs):
        selected = []
        for i in range(1, 11):
            name = kwargs.get(f"lora_name_{i}")
            if not name or name == "None":
                continue
            strength = float(kwargs.get(f"lora_strength_{i}", 1.0))
            selected.append((name, strength))

        if not selected:
            return (model, clip)

        # Build wrapper-friendly records (same path as single LoRA)
        all_model_recs, all_text_recs = [], []
        for name, strength in selected:
            path = folder_paths.get_full_path_or_raise("loras", name)
            sd = _load_lora(path)
            all_model_recs.extend(_to_qwen_records(sd, strength))
            all_text_recs.extend(_to_qwen_records(sd, strength))

        # Resolve cores
        model_core = getattr(model, "model", None) or model
        clip_core = clip or _discover_clip_from_model(model)

        # One-shot wrapper apply; no hook, no special options
        try:
            if 'update_model_with_qwen_lora' in globals() and update_model_with_qwen_lora is not None:
                update_model_with_qwen_lora(model_core, all_model_recs)
                if clip_core is not None and all_text_recs:
                    update_model_with_qwen_lora(clip_core, all_text_recs)
        except Exception as e:
            try:
                logger.exception("[Qwen-LoRA-Stack] wrapper-batched apply failed: %s", e)
            except Exception:
                pass

        return (model, clip_core)

