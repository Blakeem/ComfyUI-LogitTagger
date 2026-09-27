"""ComfyUI bootstrap for the standalone tagger harness, adapted from ContextAnchoredTileRefine.

`bootstrap()` runs before the first comfy or folder_paths import, since those names resolve
to whichever source tree sits on sys.path and folder_paths fixes its base path at import.
"""

import importlib.util
import os
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parent.parent

# The Desktop build keeps its python source apart from its models.
DEFAULT_COMFYUI_ROOT = Path(r"C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI")
MODEL_BASE_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI")

# The memory flags the ComfyUI server on this machine launches with. They change how weights load, not the math.
MEMORY_ARGS = ("--disable-pinned-memory", "--fast-disk")


def resolve_root() -> Path:
    """The ComfyUI source root, from COMFYUI_ROOT when set."""
    override = os.environ.get("COMFYUI_ROOT")
    root = Path(override).resolve() if override else DEFAULT_COMFYUI_ROOT

    if not (root / "comfy").is_dir():
        raise SystemExit(f"ComfyUI root {root} has no 'comfy' directory. Set COMFYUI_ROOT to the server source.")
    return root


def enable_dynamic_vram() -> tuple[bool, str]:
    """Bring up core's DynamicVRAM (comfy_aimdo) as main.py does, and return (enabled, why).

    Core takes its CUDA graph decode only with DynamicVRAM on, so without it a generate runs about
    5x slower than in the app. Mirrors main.py's control.init block and its later init_devices block.
    """
    import comfy_aimdo.control
    from comfy.cli_args import args, enables_dynamic_vram

    simple_vram_headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
    device_headroom = int(args.vram_headroom * 1024 ** 3)
    supported = False

    if not enables_dynamic_vram():
        return False, "the parsed ComfyUI flags turn DynamicVRAM off"
    # main.py calls control.init before anything initializes a torch device.
    comfy_aimdo.control.init(simple_vram_headroom=simple_vram_headroom, nvml_pressure=not args.disable_nvml_pressure)

    import comfy.memory_management
    import comfy.model_management as model_management
    import comfy.model_patcher

    supported = model_management.is_nvidia() or (model_management.is_amd() and model_management.rocm_version >= (7, 14))
    if not (args.enable_dynamic_vram or supported):
        return False, "the GPU is neither NVIDIA nor AMD with ROCm 7.14 or later"
    if not args.enable_dynamic_vram and model_management.torch_version_numeric < (2, 8):
        return False, f"torch {model_management.torch_version_numeric} is older than 2.8"
    if not comfy_aimdo.control.init_devices((device.index, device_headroom)
                                            for device in model_management.get_all_torch_devices()):
        return False, "comfy_aimdo reported no working install"
    comfy_aimdo.control.set_log_info()
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    return True, "comfy_aimdo initialized every torch device"


def end_node_execution() -> None:
    """The cleanup core's executor runs after every node under DynamicVRAM, at execution.py:548-553."""
    import comfy.memory_management

    if not comfy.memory_management.aimdo_enabled:
        return
    import comfy.model_management
    import comfy.model_prefetch
    import comfy_aimdo.model_vbar

    comfy.model_prefetch.cleanup_prefetch_queues()
    comfy.model_management.reset_cast_buffers()
    comfy_aimdo.model_vbar.vbars_reset_watermark_limits()


def bootstrap() -> Path:
    """Put the ComfyUI source and this pack on sys.path, parse the app's flags and enable DynamicVRAM."""
    root = resolve_root()
    spec = None
    origin = None
    saved_argv = sys.argv
    unset_flags: list[str] = []
    dynamic_vram = False
    why = ""

    for path in (str(PACK_ROOT), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    # An installed 'comfy' package, such as the one comfy-cli ships, beats the source tree
    # whatever the sys.path order.
    spec = importlib.util.find_spec("comfy.sd")
    origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    if origin is None or origin.parent != (root / "comfy").resolve():
        raise SystemExit(f"A different 'comfy' package shadows the ComfyUI source at {root}: comfy.sd is {origin}")

    # main.py sets this before native allocators load.
    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"

    # cli_args parses argv once at import, and only after enable_args_parsing, as main.py does.
    # A synthetic argv keeps the harness's own flags away from ComfyUI's parser.
    import comfy.options

    comfy.options.enable_args_parsing()
    sys.argv = [saved_argv[0], "--base-directory", str(MODEL_BASE_DIR), *MEMORY_ARGS]
    try:
        import folder_paths  # noqa: F401
    finally:
        sys.argv = saved_argv

    # An earlier comfy import would have parsed the real argv without a word.
    from comfy.cli_args import args

    unset_flags = [flag for flag in MEMORY_ARGS if not getattr(args, flag.lstrip("-").replace("-", "_"))]
    if args.base_directory != str(MODEL_BASE_DIR) or unset_flags:
        raise SystemExit(
            f"comfy.cli_args was parsed before bootstrap(): --base-directory is {args.base_directory!r}, "
            f"expected {str(MODEL_BASE_DIR)!r}, and these flags never took effect: {unset_flags}"
        )

    # main.py imports it before torch loads. It picks the app's cudaMallocAsync allocator from the parsed flags.
    import cuda_malloc  # noqa: F401

    dynamic_vram, why = enable_dynamic_vram()
    print(f"DynamicVRAM {'on' if dynamic_vram else 'OFF, so generate is not the speed the app gets'}: {why}",
          flush=True)
    return root


def load_clip(clip_name: str):
    """CLIPLoader.load_clip with type krea2 and device default."""
    import comfy.sd
    import folder_paths

    clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    return comfy.sd.load_clip(
        ckpt_paths=[clip_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.KREA2,
        model_options={},
    )
