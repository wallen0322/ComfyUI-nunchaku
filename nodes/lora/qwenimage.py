
# nodes/lora/qwenimage.py (fix16.5: functional-LoRA uses qkv weight-patch; auto-merge; minimal UI)
import os, logging, unicodedata, re
import folder_paths
from typing import Dict, Any, List, Tuple, Optional
from safetensors.torch import load_file

from ...wrappers.qwenimage import update_model_with_qwen_lora, set_qwen_lora_options

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PUNC_MAP = {"。": ".", "．": ".", "・": ".", "·": ".", "｡": ".", "﹒": "."}
WORD_MAP = {"阿尔法": "alpha", "α": "alpha"}
SUFFIX_RE = re.compile(r'\.(?:lora_A|lora_B|lora_down|lora_up)(?:\.[A-Za-z0-9_]+)?\.weight$')

def _normalize_key(k: str) -> str:
    k = unicodedata.normalize("NFKC", k)
    for a, b in PUNC_MAP.items(): k = k.replace(a, b)
    for a, b in WORD_MAP.items(): k = k.replace(a, b)
    return k

def _load_lora(path: str) -> Dict[str, Any]:
    sd = load_file(path)
    return {_normalize_key(k): v for k, v in sd.items()}

def _collect_bases(sd: Dict[str, Any]):
    bases = set()
    for k in sd.keys():
        if SUFFIX_RE.search(k):
            parts = k.split(".")
            if len(parts) >= 3 and parts[-2] == "default":
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

    qkv_last = {"to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "q_proj", "k_proj", "v_proj"}
    for base in sorted(bases):
        A = _fetch(sd,
                   f"{base}.lora_A.weight", f"{base}.lora_A.default.weight",
                   f"{base}.lora_down.weight", f"{base}.lora_down.default.weight")
        B = _fetch(sd,
                   f"{base}.lora_B.weight", f"{base}.lora_B.default.weight",
                   f"{base}.lora_up.weight", f"{base}.lora_up.default.weight")
        if A is None or B is None: continue
        alpha_t = _fetch(sd, f"{base}.alpha", f"{base}.alpha.default")
        try:
            alpha = float(alpha_t.item()) if hasattr(alpha_t, "item") else float(A.shape[0])
        except Exception:
            alpha = float(A.shape[0])

        last = base.split(".")[-1]
        parent = ".".join(base.split(".")[:-1])

        if last in qkv_last:
            if last.startswith("to_"):
                fused = parent + ".to_qkv.weight"
            elif last.startswith("add_"):
                fused = parent + ".add_qkv_proj.weight"
            else:
                fused = parent + ".W_pack.weight"
            which = "q" if last in ("to_q","add_q_proj","q_proj") else ("k" if last in ("to_k","add_k_proj","k_proj") else "v")
            recs.append(("qkv", fused, A, B, alpha, float(strength), which))
        else:
            recs.append(("linear", base + ".weight", A, B, alpha, float(strength), None))

    logger.info(f"[Qwen-LoRA] parsed bases={len(bases)}, qkv_pairs={sum(1 for r in recs if r[0]=='qkv')}, linear_pairs={sum(1 for r in recs if r[0]=='linear')}")
    return recs

def _discover_clip_from_model(model_obj) -> Optional[Any]:
    for name in ("text_encoder", "clip", "te", "t5", "qwen_text", "qwen_te", "transformer", "encoder", "cond_stage_model", "context_encoder", "text_model"):
        if hasattr(model_obj, name):
            return getattr(model_obj, name)
    inner = getattr(model_obj, "model", None)
    if inner is not None:
        return _discover_clip_from_model(inner)
    return None

class NunchakuQwenImageAutoLoRALoader:
    TITLE = "Nunchaku Qwen-Image Auto LoRA Loader"
    CATEGORY = "Nunchaku"
    FUNCTION = "load_lora"
    DESCRIPTION = "Qwen-Image LoRA loader (fix16.5): 功能LoRA的QKV走权重patch；其余走前向hook；自动合并；最小UI。"
    RETURN_TYPES = ("MODEL", "CLIP")

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

    def _apply_defaults(self, target, recs, role, functional=False, boost_qkv=False):
        if target is None or not recs:
            return
        try:
            set_qwen_lora_options(target, functional=functional, boost_qkv=bool(boost_qkv))
        except Exception:
            pass
        w = getattr(target, "model", None) or getattr(target, "text_encoder", None) or target
        update_model_with_qwen_lora(w, recs)
        logger.info(f"[Qwen-LoRA] applied to {type(w).__name__} ({role})")

    def load_lora(self, lora_name: str, model_strength: float, text_strength: float, model=None, clip=None):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = _load_lora(lora_path)

        recs_model = _to_qwen_records(sd, model_strength)
        recs_text  = _to_qwen_records(sd, text_strength)

        lower_name = lora_name.lower()
        is_func = any(k in lower_name for k in ("light", "lcm", "consist"))

        if is_func:
            logger.info("[Qwen-LoRA] functional LoRA detected -> QKV weight-patch mode + slight QKV boost (steps≈4-8, cfg≈1.0).")
        else:
            logger.info("[Qwen-LoRA] normal/style LoRA -> forward-hook mode.")

        if model is not None:
            logger.info(f"[Qwen-LoRA] MODEL appended {len(recs_model)} records")
            self._apply_defaults(model, recs_model, role="MODEL", functional=is_func, boost_qkv=is_func)

            if clip is None:
                auto_clip = _discover_clip_from_model(model)
                if auto_clip is not None:
                    logger.info("[Qwen-LoRA] auto-discovered TE from MODEL")
                    logger.info(f"[Qwen-LoRA] AUTO-CLIP appended {len(recs_text)} records")
                    self._apply_defaults(auto_clip, recs_text, role="AUTO-CLIP", functional=False, boost_qkv=False)

        if clip is not None:
            logger.info(f"[Qwen-LoRA] CLIP appended {len(recs_text)} records")
            self._apply_defaults(clip, recs_text, role="CLIP", functional=False, boost_qkv=False)

        return (model, clip)
