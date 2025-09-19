
import os, logging, unicodedata, re
import folder_paths
from typing import Dict, Any, List, Tuple, Set
from safetensors.torch import load_file

from ...wrappers.qwenimage import update_model_with_qwen_lora

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PUNC_MAP = {"。": ".", "．": ".", "・": ".", "·": ".", "｡": ".", "﹒": "."}
WORD_MAP = {"阿尔法": "alpha", "α": "alpha"}
SUFFIX_RE = re.compile(r'\.(?:lora_A|lora_B|lora_down|lora_up)(?:\.[A-Za-z0-9_]+)?\.weight$')

def _normalize_key(k: str) -> str:
    import unicodedata as U
    k = U.normalize("NFKC", k)
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

    qkv_last = {"to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"}

    qkv_cnt = 0; lin_cnt = 0
    for base in sorted(bases):
        A = _fetch(sd, f"{base}.lora_A.weight", f"{base}.lora_A.default.weight",
                        f"{base}.lora_down.weight", f"{base}.lora_down.default.weight")
        B = _fetch(sd, f"{base}.lora_B.weight", f"{base}.lora_B.default.weight",
                        f"{base}.lora_up.weight", f"{base}.lora_up.default.weight")
        if A is None or B is None: continue
        alpha_t = _fetch(sd, f"{base}.alpha", f"{base}.alpha.default")
        try: alpha = float(alpha_t.item()) if hasattr(alpha_t, "item") else float(A.shape[0])
        except Exception: alpha = float(A.shape[0])

        last = base.split(".")[-1]
        parent = ".".join(base.split(".")[:-1])
        if last in qkv_last:
            fused = parent + ("." + ("to_qkv" if last.startswith("to_") else "add_qkv_proj") + ".weight")
            which = "q" if last in ("to_q","add_q_proj") else ("k" if last in ("to_k","add_k_proj") else "v")
            recs.append(("qkv", fused, A, B, alpha, float(strength), which)); qkv_cnt += 1
        else:
            recs.append(("linear", base + ".weight", A, B, alpha, float(strength), None)); lin_cnt += 1

    logger.info(f"[Qwen-LoRA] parsed bases={len(bases)}, qkv_pairs={qkv_cnt}, linear_pairs={lin_cnt}")
    return recs

class NunchakuQwenImageAutoLoRALoader:
    TITLE = "Nunchaku Qwen-Image Auto LoRA Loader"
    CATEGORY = "Nunchaku"
    FUNCTION = "load_lora"
    DESCRIPTION = "Parse Qwen-Image LoRA -> append to wrappers and APPLY NOW via low-rank hooks."
    RETURN_TYPES = ("MODEL", "CLIP")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -8.0, "max": 8.0, "step": 0.01}),
            },
            "optional": {"model": ("MODEL", {}), "clip": ("CLIP", {})}
        }

    def _append_and_apply(self, wrapper_obj, recs):
        if wrapper_obj is None or not recs:
            return
        w = getattr(wrapper_obj, "model", None) or getattr(wrapper_obj, "text_encoder", None) or wrapper_obj
        if not hasattr(w, "qwen_loras"): w.qwen_loras = []
        w.qwen_loras.extend(recs)
        try:
            update_model_with_qwen_lora(w, w.qwen_loras)
            logger.info("[Qwen-LoRA] applied to %s", type(w).__name__)
        except Exception as e:
            logger.exception("[Qwen-LoRA] apply failed on %s: %s", type(w).__name__, e)

    def load_lora(self, lora_name: str, lora_strength: float, model=None, clip=None):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = _load_lora(lora_path)
        recs = _to_qwen_records(sd, lora_strength)

        if model is not None:
            logger.info(f"[Qwen-LoRA] MODEL appended {len(recs)} records")
            self._append_and_apply(model, recs)

        if clip is not None:
            logger.info(f"[Qwen-LoRA] CLIP appended {len(recs)} records")
            self._append_and_apply(clip, recs)

        return (model, clip)
