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


def resolve_root() -> Path:
    """The ComfyUI source root, from COMFYUI_ROOT when set."""
    override = os.environ.get("COMFYUI_ROOT")
    root = Path(override).resolve() if override else DEFAULT_COMFYUI_ROOT

    if not (root / "comfy").is_dir():
        raise SystemExit(f"ComfyUI root {root} has no 'comfy' directory. Set COMFYUI_ROOT to the server source.")
    return root


def bootstrap() -> Path:
    """Put the ComfyUI source and this pack on sys.path and point folder_paths at the model base."""
    root = resolve_root()
    spec = None
    origin = None
    saved_argv = sys.argv

    for path in (str(PACK_ROOT), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    # An installed 'comfy' package, such as the one comfy-cli ships, beats the source tree
    # whatever the sys.path order.
    spec = importlib.util.find_spec("comfy.sd")
    origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    if origin is None or origin.parent != (root / "comfy").resolve():
        raise SystemExit(f"A different 'comfy' package shadows the ComfyUI source at {root}: comfy.sd is {origin}")

    # cli_args parses argv once at import, and only after enable_args_parsing, as main.py does.
    # A synthetic argv keeps the harness's own flags away from ComfyUI's parser.
    import comfy.options

    comfy.options.enable_args_parsing()
    sys.argv = [saved_argv[0], "--base-directory", str(MODEL_BASE_DIR)]
    try:
        import folder_paths  # noqa: F401
    finally:
        sys.argv = saved_argv

    # An earlier comfy import would have parsed the real argv without a word.
    from comfy.cli_args import args

    if args.base_directory != str(MODEL_BASE_DIR):
        raise SystemExit(
            f"comfy.cli_args was parsed before bootstrap(): --base-directory is {args.base_directory!r}, "
            f"expected {str(MODEL_BASE_DIR)!r}"
        )
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
