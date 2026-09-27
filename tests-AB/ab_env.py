"""This machine's paths and flags for the library's harness bootstrap, comfy_env.

`bootstrap()` runs before the first comfy or folder_paths import, since those names resolve
to whichever source tree sits on sys.path and folder_paths fixes its base path at import.
"""

import os
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parent.parent

# The Desktop build keeps its python source apart from its models.
DEFAULT_COMFYUI_ROOT = Path(r"C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI")
MODEL_BASE_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI")

# The memory flags the ComfyUI server on this machine launches with. They change how weights load, not the math.
MEMORY_ARGS = ("--disable-pinned-memory", "--fast-disk")

# comfy_env ships in the library's repository and in no install. Appended, so its neighbours never shadow a module.
LIBRARY_HARNESS_DIR = Path(r"E:\logit-classifier\tests-AB")

if not (LIBRARY_HARNESS_DIR / "comfy_env.py").is_file():
    raise SystemExit(f"{LIBRARY_HARNESS_DIR} holds no comfy_env.py. Point LIBRARY_HARNESS_DIR at the "
                     "logit-classifier repository's tests-AB folder.")
if str(LIBRARY_HARNESS_DIR) not in sys.path:
    sys.path.append(str(LIBRARY_HARNESS_DIR))

import comfy_env  # noqa: E402


def resolve_root() -> Path:
    """The ComfyUI source root, from COMFYUI_ROOT when set."""
    override = os.environ.get("COMFYUI_ROOT")
    root = Path(override).resolve() if override else DEFAULT_COMFYUI_ROOT

    if not (root / "comfy").is_dir():
        raise SystemExit(f"ComfyUI root {root} has no 'comfy' directory. Set COMFYUI_ROOT to the server source.")
    return root


def bootstrap() -> Path:
    """Put this pack on sys.path, then import comfy from the ComfyUI source with the app's flags and DynamicVRAM."""
    root = resolve_root()

    if str(PACK_ROOT) not in sys.path:
        sys.path.insert(0, str(PACK_ROOT))
    comfy_env.bootstrap(root, MODEL_BASE_DIR, MEMORY_ARGS)
    return root


def end_node_execution() -> None:
    """The cleanup core's executor runs after every node, for a harness that runs several node executions."""
    comfy_env.end_node_execution()


def load_clip(clip_name: str):
    """CLIPLoader.load_clip with type krea2 and device default."""
    return comfy_env.load_clip(clip_name, clip_type="krea2")
