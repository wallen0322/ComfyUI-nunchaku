
import logging
import os

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from .utils import get_package_version, get_plugin_version, supported_versions

nunchaku_full_version = get_package_version("nunchaku").split("+")[0].strip()
logger.info(f"Nunchaku version: {nunchaku_full_version}")
logger.info(f"ComfyUI-nunchaku version: {get_plugin_version()}")

nunchaku_version = nunchaku_full_version.split("+")[0].strip()
nunchaku_major_minor_patch_version = ".".join(nunchaku_version.split(".")[:3])
if f"v{nunchaku_major_minor_patch_version}" not in supported_versions:
    logger.warning(
        f"ComfyUI-nunchaku {get_plugin_version()} may not match nunchaku {nunchaku_full_version}. "
        f"Supported: {supported_versions}."
    )

NODE_CLASS_MAPPINGS = {}

# --- keep existing registrations (Flux/Qwen loaders, etc.) if present in your original file ---
try:
    from .nodes.models.flux import NunchakuFluxDiTLoader
    NODE_CLASS_MAPPINGS["NunchakuFluxDiTLoader"] = NunchakuFluxDiTLoader
except Exception:
    logger.exception("NunchakuFluxDiTLoader import failed")

try:
    from .nodes.models.qwenimage import NunchakuQwenImageDiTLoader
    NODE_CLASS_MAPPINGS["NunchakuQwenImageDiTLoader"] = NunchakuQwenImageDiTLoader
except Exception:
    logger.exception("NunchakuQwenImageDiTLoader import failed")

try:
    from .nodes.lora.flux import NunchakuFluxLoraLoader, NunchakuFluxLoraStack
    NODE_CLASS_MAPPINGS["NunchakuFluxLoraLoader"] = NunchakuFluxLoraLoader
    NODE_CLASS_MAPPINGS["NunchakuFluxLoraStack"] = NunchakuFluxLoraStack
except Exception:
    logger.exception("Flux LoRA imports failed")

# === Only one new node (merged): Qwen-Image Auto LoRA (in qwenimage.py) ===
try:
    from .nodes.lora.qwenimage import NunchakuQwenImageAutoLoRALoader
    NODE_CLASS_MAPPINGS["NunchakuQwenImageAutoLoRALoader"] = NunchakuQwenImageAutoLoRALoader
except Exception:
    logger.exception("Qwen-Image Auto LoRA import failed")

try:
    from .nodes.lora.qwenimage import NunchakuQwenImageLoRALoader, NunchakuQwenImageLoRAStack
    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoRALoader"] = NunchakuQwenImageLoRALoader
    NODE_CLASS_MAPPINGS["NunchakuQwenImageLoRAStack"] = NunchakuQwenImageLoRAStack
except Exception:
    logger.exception("Qwen-Image LoRA imports failed")


# keep other original registrations if needed ...
try:
    from .nodes.models.text_encoder import NunchakuTextEncoderLoader, NunchakuTextEncoderLoaderV2
    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoader"] = NunchakuTextEncoderLoader
    NODE_CLASS_MAPPINGS["NunchakuTextEncoderLoaderV2"] = NunchakuTextEncoderLoaderV2
except Exception:
    pass

try:
    from .nodes.preprocessors.depth import FluxDepthPreprocessor
    NODE_CLASS_MAPPINGS["NunchakuDepthPreprocessor"] = FluxDepthPreprocessor
except Exception:
    pass

try:
    from .nodes.models.pulid import (
        NunchakuFluxPuLIDApplyV2,
        NunchakuPulidApply,
        NunchakuPulidLoader,
        NunchakuPuLIDLoaderV2,
    )
    NODE_CLASS_MAPPINGS["NunchakuPulidApply"] = NunchakuPulidApply
    NODE_CLASS_MAPPINGS["NunchakuPulidLoader"] = NunchakuPulidLoader
    NODE_CLASS_MAPPINGS["NunchakuPuLIDLoaderV2"] = NunchakuPuLIDLoaderV2
    NODE_CLASS_MAPPINGS["NunchakuFluxPuLIDApplyV2"] = NunchakuFluxPuLIDApplyV2
except Exception:
    pass

try:
    from .nodes.models.ipadapter import NunchakuFluxIPAdapterApply, NunchakuIPAdapterLoader
    NODE_CLASS_MAPPINGS["NunchakuFluxIPAdapterApply"] = NunchakuFluxIPAdapterApply
    NODE_CLASS_MAPPINGS["NunchakuIPAdapterLoader"] = NunchakuIPAdapterLoader
except Exception:
    pass

try:
    from .nodes.tools.merge_safetensors import NunchakuModelMerger
    NODE_CLASS_MAPPINGS["NunchakuModelMerger"] = NunchakuModelMerger
except Exception:
    pass

try:
    from .nodes.tools.installers import NunchakuWheelInstaller
    NODE_CLASS_MAPPINGS["NunchakuWheelInstaller"] = NunchakuWheelInstaller
except Exception:
    pass

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
